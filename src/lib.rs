#![deny(missing_docs)]

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

    use std::ops::ControlFlow;

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn random_points(
            xs in prop::collection::vec(0.0..=1.0, 100),
            ys in prop::collection::vec(0.0..=1.0, 100),
            cxs in prop::collection::vec(0.0..=1.0, 10),
            cys in prop::collection::vec(0.0..=1.0, 10),
            ds in prop::collection::vec(0.0..=1.0, 10),
        ) {
            #[derive(Debug, PartialEq)]
            struct RandomObject([f64; 2]);

            impl Object for RandomObject {
                type Point = [f64; 2];

                fn position(&self) -> &Self::Point {
                    &self.0
                }
            }

            let index = KdTree::new(
                xs.into_iter()
                    .zip(ys)
                    .map(|(x, y)| RandomObject([x, y]))
                    .collect(),
            );

            for ((cx, cy), d) in cxs.into_iter().zip(cys).zip(ds) {
                let query = WithinDistance::new([cx, cy], d);

                let mut results1 = Vec::new();
                index.look_up(&query, |object| {
                    results1.push(object);
                    ControlFlow::Continue(())
                });

                let mut results2 = index
                    .iter()
                    .filter(|object| query.test(object.position()))
                    .collect::<Vec<_>>();

                let cmp_objs = |lhs: &&RandomObject, rhs: &&RandomObject| lhs.0.partial_cmp(&rhs.0).unwrap();
                results1.sort_unstable_by(cmp_objs);
                results2.sort_unstable_by(cmp_objs);

                assert_eq!(results1, results2);

                let target = [cx, cy];

                let result1 = index.nearest(&target).unwrap();

                let result2 = index
                    .iter()
                    .min_by(|lhs, rhs| {
                        let lhs = lhs.0.distance_2(&target);
                        let rhs = rhs.0.distance_2(&target);

                        lhs.partial_cmp(&rhs).unwrap()
                    })
                    .unwrap();

                assert_eq!(result1, result2);
            }
        }
    }
}
