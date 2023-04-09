use std::ops::ControlFlow;

use crate::{contains, split, KdTree, Object, Point};

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
        self.center.distance_2(position) <= self.distance_2
    }
}

impl<O: Object> KdTree<O> {
    /// TODO
    pub fn look_up<'a, Q: Query<O::Point>, V: FnMut(&'a O) -> ControlFlow<()>>(
        &'a self,
        query: &Q,
        visitor: V,
    ) {
        if !self.objects.is_empty() {
            look_up(&mut LookUpArgs { query, visitor }, &self.objects, 0);
        }
    }
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
