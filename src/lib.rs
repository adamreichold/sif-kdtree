#![deny(missing_docs, missing_debug_implementations)]

//! A simple library implementing an immutable, flat representation of a [k-d tree](https://en.wikipedia.org/wiki/K-d_tree)
//!
//! The library supports arbitrary spatial queries via the [`Query`] trait and nearest neighbour search.
//! Its implementation is simple as the objects in the index are fixed after construction.
//! This also enables a flat and thereby cache-friendly memory layout.
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
//!     [
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
//!     ]
//!     .into(),
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

mod look_up;
mod nearest;
mod sort;

pub use look_up::{Query, WithinBoundingBox, WithinDistance};

use std::ops::Deref;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Defines a [finite-dimensional][Self::DIM] real space in terms of [coordinate values][Self::coord] along a chosen set of axes
pub trait Point {
    /// The dimension of the underlying real space
    const DIM: usize;

    /// Access the coordinate value of the point along the given `axis`
    fn coord(&self, axis: usize) -> f64;

    /// Return the squared distance between `self` and `other`.
    ///
    /// This is called during nearest neighbour search and hence only the relation between two distance values is required so that computing square roots can be avoided.
    fn distance_2(&self, other: &Self) -> f64;
}

/// `N`-dimensional real space using [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)
impl<const N: usize> Point for [f64; N] {
    const DIM: usize = N;

    fn coord(&self, axis: usize) -> f64 {
        self[axis]
    }

    fn distance_2(&self, other: &Self) -> f64 {
        (0..N).map(|axis| (self[axis] - other[axis]).powi(2)).sum()
    }
}

/// Defines the objects which can be organized in a [`KdTree`] by positioning them in a real space defined via the [`Point`] trait
pub trait Object {
    /// The [`Point`] implementation used to represent the [position][`Self::position`] of these objects.
    type Point: Point;

    /// Return the position associated with this object.
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
#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct KdTree<O> {
    objects: Box<[O]>,
}

impl<O> Deref for KdTree<O> {
    type Target = [O];

    fn deref(&self) -> &Self::Target {
        &self.objects
    }
}

impl<O> AsRef<[O]> for KdTree<O> {
    fn as_ref(&self) -> &[O] {
        &self.objects
    }
}

fn split<O>(objects: &[O]) -> (&[O], &O, &[O]) {
    assert!(!objects.is_empty());

    let mid = objects.len() / 2;

    unsafe {
        (
            objects.get_unchecked(..mid),
            objects.get_unchecked(mid),
            objects.get_unchecked(mid + 1..),
        )
    }
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

    pub fn random_points(len: usize) -> impl Strategy<Value = Vec<[f64; 2]>> {
        (vec(0.0..=1.0, len), vec(0.0..=1.0, len))
            .prop_map(|(x, y)| x.into_iter().zip(y).map(|(x, y)| [x, y]).collect())
    }

    #[derive(Debug, PartialEq)]
    pub struct RandomObject(pub [f64; 2]);

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
        type Point = [f64; 2];

        fn position(&self) -> &Self::Point {
            &self.0
        }
    }

    pub fn random_objects(len: usize) -> impl Strategy<Value = Box<[RandomObject]>> {
        random_points(len).prop_map(|points| points.into_iter().map(RandomObject).collect())
    }
}
