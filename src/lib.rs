#![forbid(unsafe_code)]
#![deny(missing_docs, missing_debug_implementations)]

//! A simple library implementing an immutable, flat representation of a [k-d tree](https://en.wikipedia.org/wiki/K-d_tree)
//!
//! The library supports arbitrary spatial queries via the [`Query`] trait and nearest neighbour search.
//! Its implementation is simple as the objects in the index are fixed after construction.
//! This also enables a flat and thereby cache-friendly memory layout which can be backed by memory maps.
//!
//! The library provides optional integration with [rayon] for parallel construction and queries and [serde] for (de-)serialization of the trees.
//!
//! # Example
//!
//! ```
//! use std::ops::ControlFlow;
//!
//! use sif_kdtree::{KdTree, Object, WithinDistance};
//!
//! struct Something(usize, [f64; 2]);
//!
//! impl Object for Something {
//!     type Point = [f64; 2];
//!
//!     fn position(&self) -> &Self::Point {
//!         &self.1
//!     }
//! }
//!
//! let index = KdTree::new(
//!     vec![
//!         Something(0, [-0.4, -3.3]),
//!         Something(1, [-4.5, -1.8]),
//!         Something(2, [0.7, 2.0]),
//!         Something(3, [1.7, 1.5]),
//!         Something(4, [-1.3, 2.3]),
//!         Something(5, [2.2, 1.0]),
//!         Something(6, [-3.7, 3.8]),
//!         Something(7, [-3.2, -0.1]),
//!         Something(8, [1.4, 2.7]),
//!         Something(9, [3.1, -0.0]),
//!         Something(10, [4.3, 0.8]),
//!         Something(11, [3.9, -3.3]),
//!         Something(12, [0.4, -3.2]),
//!     ],
//! );
//!
//! let mut close_by = Vec::new();
//!
//! index.look_up(&WithinDistance::new([0., 0.], 3.), |thing| {
//!     close_by.push(thing.0);
//!
//!     ControlFlow::Continue(())
//! });
//!
//! assert_eq!(close_by, [2, 4, 5, 3]);
//!
//! let closest = index.nearest(&[0., 0.]).unwrap().0;
//!
//! assert_eq!(closest, 2);
//! ```
//!
//! The [`KdTree`] data structure is generic over its backing storage as long as it can be converted into a slice via the [`AsRef`] trait.
//! This can for instance be used to memory map k-d trees from persistent storage.
//!
//! ```no_run
//! # fn main() -> std::io::Result<()> {
//! use std::fs::File;
//! use std::mem::size_of;
//! use std::slice::from_raw_parts;
//!
//! use memmap2::Mmap;
//!
//! use sif_kdtree::{KdTree, Object};
//!
//! #[derive(Clone, Copy)]
//! struct Point([f64; 3]);
//!
//! impl Object for Point {
//!     type Point = [f64; 3];
//!
//!     fn position(&self) -> &Self::Point {
//!         &self.0
//!     }
//! }
//!
//! let file = File::open("index.bin")?;
//! let map = unsafe { Mmap::map(&file)? };
//!
//! struct PointCloud(Mmap);
//!
//! impl AsRef<[Point]> for PointCloud {
//!     fn as_ref(&self) -> &[Point] {
//!         let ptr = self.0.as_ptr().cast();
//!         let len = self.0.len() / size_of::<Point>();
//!
//!         unsafe { from_raw_parts(ptr, len) }
//!     }
//! }
//!
//! let index = KdTree::new_unchecked(PointCloud(map));
//! # Ok(()) }
//! ```

mod look_up;
mod nearest;
mod sort;

pub use look_up::{Query, WithinBoundingBox, WithinDistance};

use std::marker::PhantomData;
use std::ops::Deref;

use num_traits::Num;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Defines a [finite-dimensional][Self::DIM] space in terms of [coordinate values][Self::coord] along a chosen set of axes
pub trait Point {
    /// The dimension of the underlying space
    const DIM: usize;

    /// The type of the coordinate values
    type Coord: Num + Copy + PartialOrd;

    /// Access the coordinate value of the point along the given `axis`
    fn coord(&self, axis: usize) -> Self::Coord;
}

/// Extends the [`Point`] trait by a distance metric required for nearest neighbour search
pub trait Distance: Point {
    /// Return the squared distance between `self` and `other`
    ///
    /// This is called during nearest neighbour search and hence only the relation between two distance values is required so that computing square roots can be avoided.
    fn distance_2(&self, other: &Self) -> Self::Coord;
}

/// `N`-dimensional space using [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)
impl<T, const N: usize> Point for [T; N]
where
    T: Num + Copy + PartialOrd,
{
    const DIM: usize = N;

    type Coord = T;

    fn coord(&self, axis: usize) -> Self::Coord {
        self[axis]
    }
}

impl<T, const N: usize> Distance for [T; N]
where
    T: Num + Copy + PartialOrd,
{
    fn distance_2(&self, other: &Self) -> Self::Coord {
        (0..N).fold(T::zero(), |res, axis| {
            let diff = self[axis] - other[axis];

            res + diff * diff
        })
    }
}

/// Defines the objects which can be organized in a [`KdTree`] by positioning them in the vector space defined via the [`Point`] trait
pub trait Object {
    /// The [`Point`] implementation used to represent the [position][`Self::position`] of these objects
    type Point: Point;

    /// Return the position associated with this object
    ///
    /// Note that calling this method is assumed to be cheap, returning a reference to a point stored in the interior of the object.
    fn position(&self) -> &Self::Point;
}

/// An immutable, flat representation of a [k-d tree](https://en.wikipedia.org/wiki/K-d_tree)
///
/// Accelerates spatial queries and nearest neighbour search by sorting the objects according to the coordinate values of their positions.
///
/// Note that this tree dereferences to and deserializes as a slice of objects.
/// Modifying object positions through interior mutability or deserializing a modified sequence is safe but will lead to incorrect results.
#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct KdTree<O, S = Box<[O]>>
where
    S: AsRef<[O]>,
{
    objects: S,
    _marker: PhantomData<O>,
}

impl<O, S> KdTree<O, S>
where
    O: Object,
    S: AsRef<[O]>,
{
    /// Interprets the given `objects` as a tree
    ///
    /// Supplying `objects` which are not actually sorted as a k-d tree is safe but will lead to incorrect results.
    pub fn new_unchecked(objects: S) -> Self {
        Self {
            objects,
            _marker: PhantomData,
        }
    }
}

impl<O, S> Deref for KdTree<O, S>
where
    S: AsRef<[O]>,
{
    type Target = [O];

    fn deref(&self) -> &Self::Target {
        self.objects.as_ref()
    }
}

impl<O, S> AsRef<[O]> for KdTree<O, S>
where
    S: AsRef<[O]>,
{
    fn as_ref(&self) -> &[O] {
        self.objects.as_ref()
    }
}

fn split<O>(objects: &[O]) -> (&[O], &O, &[O]) {
    let (left, objects) = objects.split_at(objects.len() / 2);
    let (mid, right) = objects.split_first().unwrap();

    (left, mid, right)
}

fn contains<P>(aabb: &(P, P), position: &P) -> bool
where
    P: Point,
{
    (0..P::DIM).all(|axis| {
        aabb.0.coord(axis) <= position.coord(axis) && position.coord(axis) <= aabb.1.coord(axis)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cmp::Ordering;

    use proptest::{collection::vec, strategy::Strategy};

    pub fn random_points(len: usize) -> impl Strategy<Value = Vec<[f32; 2]>> {
        (vec(0.0_f32..=1.0, len), vec(0.0_f32..=1.0, len))
            .prop_map(|(x, y)| x.into_iter().zip(y).map(|(x, y)| [x, y]).collect())
    }

    #[derive(Debug, PartialEq)]
    pub struct RandomObject(pub [f32; 2]);

    impl Eq for RandomObject {}

    impl PartialOrd for RandomObject {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for RandomObject {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.partial_cmp(&other.0).unwrap()
        }
    }

    impl Object for RandomObject {
        type Point = [f32; 2];

        fn position(&self) -> &Self::Point {
            &self.0
        }
    }

    pub fn random_objects(len: usize) -> impl Strategy<Value = Box<[RandomObject]>> {
        random_points(len).prop_map(|points| points.into_iter().map(RandomObject).collect())
    }
}
