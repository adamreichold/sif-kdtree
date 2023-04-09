#![deny(missing_docs, missing_debug_implementations)]

//! TODO

mod look_up;
mod nearest;
mod sort;

pub use look_up::{Query, WithinBoundingBox, WithinDistance};

use std::ops::Deref;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// TODO
pub trait Point: Sized {
    /// TODO
    const DIM: usize;
    /// TODO
    fn coord(&self, axis: usize) -> f64;
    /// TODO
    fn distance_2(&self, other: &Self) -> f64;
}

impl<const N: usize> Point for [f64; N] {
    const DIM: usize = N;

    fn coord(&self, axis: usize) -> f64 {
        self[axis]
    }

    fn distance_2(&self, other: &Self) -> f64 {
        (0..N).map(|i| (self[i] - other[i]).powi(2)).sum()
    }
}

/// TODO
pub trait Object: Send {
    /// TODO
    type Point: Point;
    /// TODO
    fn position(&self) -> &Self::Point;
}

/// TODO
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KdTree<O> {
    objects: Box<[O]>,
}

impl<O> Deref for KdTree<O> {
    type Target = [O];

    fn deref(&self) -> &Self::Target {
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

fn contains<P: Point>(aabb: &(P, P), position: &P) -> bool {
    for axis in 0..P::DIM {
        if aabb.0.coord(axis) > position.coord(axis) || aabb.1.coord(axis) < position.coord(axis) {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cmp::Ordering;

    use proptest::{collection::vec, strategy::Strategy};

    pub fn random_points(len: usize) -> impl Strategy<Value = Vec<[f64; 2]>> {
        (vec(0.0..=1.0, len), vec(0.0..=1.0, len))
            .prop_map(|(xs, ys)| xs.into_iter().zip(ys).map(|(x, y)| [x, y]).collect())
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
