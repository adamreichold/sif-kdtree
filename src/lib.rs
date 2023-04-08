#![deny(missing_docs)]

//! TODO

use std::ops::{ControlFlow, Deref};

#[cfg(feature = "rayon")]
use rayon::join;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// TODO
pub trait Point: Sized {
    /// TODO
    const DIM: usize;
    /// TODO
    fn coord(&self, axis: usize) -> f64;
}

impl<const N: usize> Point for [f64; N] {
    const DIM: usize = N;

    fn coord(&self, axis: usize) -> f64 {
        self[axis]
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
pub trait Query<P: Point> {
    /// TODO
    fn aabb(&self) -> &(P, P);
    /// TODO
    fn test(&self, position: &P) -> bool;
}

/// TODO
pub struct WithinBoundingBox<const N: usize> {
    aabb: ([f64; N], [f64; N]),
}

impl<const N: usize> WithinBoundingBox<N> {
    /// TODO
    pub fn new(lower: [f64; N], upper: [f64; N]) -> Self {
        Self {
            aabb: (lower, upper),
        }
    }
}

impl<const N: usize> Query<[f64; N]> for WithinBoundingBox<N> {
    fn aabb(&self) -> &([f64; N], [f64; N]) {
        &self.aabb
    }

    fn test(&self, _position: &[f64; N]) -> bool {
        true
    }
}

/// TODO
pub struct WithinDistance<const N: usize> {
    aabb: ([f64; N], [f64; N]),
    center: [f64; N],
    distance_2: f64,
}

impl<const N: usize> WithinDistance<N> {
    /// TODO
    pub fn new(center: [f64; N], distance: f64) -> Self {
        Self {
            aabb: (
                center.map(|coord| coord - distance),
                center.map(|coord| coord + distance),
            ),
            center,
            distance_2: distance.powi(2),
        }
    }
}

impl<const N: usize> Query<[f64; N]> for WithinDistance<N> {
    fn aabb(&self) -> &([f64; N], [f64; N]) {
        &self.aabb
    }

    fn test(&self, position: &[f64; N]) -> bool {
        let distance_2 = (0..N)
            .map(|i| (position[i] - self.center[i]).powi(2))
            .sum::<f64>();

        distance_2 <= self.distance_2
    }
}

/// TODO
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KdTree<O> {
    objects: Box<[O]>,
}

impl<O: Object> KdTree<O> {
    /// TODO
    pub fn new(mut objects: Box<[O]>) -> Self {
        sort(&mut objects, 0);

        Self { objects }
    }

    #[cfg(feature = "rayon")]
    /// TODO
    pub fn par_new(mut objects: Box<[O]>) -> Self {
        par_sort(&mut objects, 0);

        Self { objects }
    }

    /// TODO
    pub fn look_up<'a, Q: Query<O::Point>, V: FnMut(&'a O) -> ControlFlow<()>>(
        &'a self,
        query: &Q,
        visitor: V,
    ) {
        if self.objects.is_empty() {
            return;
        }

        look_up(&mut LookUpArgs { query, visitor }, &self.objects, 0);
    }
}

impl<O> Deref for KdTree<O> {
    type Target = [O];

    fn deref(&self) -> &Self::Target {
        &self.objects
    }
}

fn sort<O: Object>(objects: &mut [O], axis: usize) {
    if objects.len() <= 1 {
        return;
    }

    let (left, right, next_axis) = sort_axis(objects, axis);

    sort(left, next_axis);
    sort(right, next_axis);
}

#[cfg(feature = "rayon")]
fn par_sort<O: Object>(objects: &mut [O], axis: usize) {
    if objects.len() <= 1 {
        return;
    }

    let (left, right, next_axis) = sort_axis(objects, axis);

    join(|| sort(left, next_axis), || sort(right, next_axis));
}

fn sort_axis<O: Object>(objects: &mut [O], axis: usize) -> (&mut [O], &mut [O], usize) {
    let mid = objects.len() / 2;

    let (left, _, right) = objects.select_nth_unstable_by(mid, |lhs, rhs| {
        let lhs = lhs.position().coord(axis);
        let rhs = rhs.position().coord(axis);

        lhs.partial_cmp(&rhs).unwrap()
    });

    let next_axis = (axis + 1) % O::Point::DIM;

    (left, right, next_axis)
}

struct LookUpArgs<'a, Q, V> {
    query: &'a Q,
    visitor: V,
}

fn look_up<'a, O: Object, Q: Query<O::Point>, V: FnMut(&'a O) -> ControlFlow<()>>(
    args: &mut LookUpArgs<Q, V>,
    mut objects: &'a [O],
    mut axis: usize,
) -> ControlFlow<()> {
    loop {
        let (left, object, right) = split(objects);

        let position = object.position();

        if contains(args.query.aabb(), position) && args.query.test(position) {
            (args.visitor)(object)?;
        }

        let next_axis = (axis + 1) % O::Point::DIM;

        if !left.is_empty() && args.query.aabb().0.coord(axis) <= position.coord(axis) {
            look_up(args, left, next_axis)?;
        }

        if right.is_empty() || args.query.aabb().1.coord(axis) < position.coord(axis) {
            return ControlFlow::Continue(());
        }

        objects = right;
        axis = next_axis;
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

            let index = KdTree::new(xs.into_iter().zip(ys).map(|(x, y)| RandomObject([x, y])).collect());

            for ((cx, cy), d) in cxs.into_iter().zip(cys).zip(ds) {
                let query = WithinDistance::new([cx, cy], d);

                let mut results1 = Vec::new();
                index.look_up(&query, |object| {
                    results1.push(object);
                    ControlFlow::Continue(())
                });

                let mut results2 = index.iter().filter(|object| {
                    (object.0[0] - cx).powi(2) + (object.0[1] - cy).powi(2) <= d.powi(2)
                }).collect::<Vec<_>>();

                let cmp_objs = |lhs: &&RandomObject, rhs: &&RandomObject| lhs.0.partial_cmp(&rhs.0).unwrap();
                results1.sort_unstable_by(cmp_objs);
                results2.sort_unstable_by(cmp_objs);

                assert_eq!(results1, results2);
            }
        }
    }
}
