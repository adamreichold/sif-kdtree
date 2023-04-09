use std::ops::ControlFlow;

#[cfg(feature = "rayon")]
use rayon::join;

use crate::{contains, split, KdTree, Object, Point};

/// TODO
pub trait Query<P: Point> {
    /// TODO
    fn aabb(&self) -> &(P, P);
    /// TODO
    fn test(&self, position: &P) -> bool;
}

/// TODO
#[derive(Debug)]
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
#[derive(Debug)]
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

    #[cfg(feature = "rayon")]
    /// TODO
    pub fn par_look_up<'a, Q: Query<O::Point> + Sync, V: Fn(&'a O) + Sync>(
        &'a self,
        query: &Q,
        visitor: V,
    ) where
        O: Sync,
        O::Point: Sync,
    {
        if !self.objects.is_empty() {
            par_look_up(&LookUpArgs { query, visitor }, &self.objects, 0);
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

#[cfg(feature = "rayon")]
fn par_look_up<'a, O: Object + Sync, Q: Query<O::Point> + Sync, V: Fn(&'a O) + Sync>(
    args: &LookUpArgs<Q, V>,
    objects: &'a [O],
    axis: usize,
) where
    O::Point: Sync,
{
    let (left, object, right) = split(objects);

    let position = object.position();

    if contains(args.query.aabb(), position) && args.query.test(position) {
        (args.visitor)(object);
    }

    let next_axis = (axis + 1) % O::Point::DIM;

    join(
        || {
            if !left.is_empty() && args.query.aabb().0.coord(axis) <= position.coord(axis) {
                par_look_up(args, left, next_axis);
            }
        },
        || {
            if !right.is_empty() && args.query.aabb().1.coord(axis) >= position.coord(axis) {
                par_look_up(args, right, next_axis);
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rayon")]
    use std::sync::Mutex;

    use proptest::{collection::vec, strategy::Strategy, test_runner::TestRunner};

    use crate::tests::{random_objects, random_points};

    pub fn random_queries(len: usize) -> impl Strategy<Value = Vec<WithinDistance<2>>> {
        (random_points(len), vec(0.0..=1.0, len)).prop_map(|(centers, distances)| {
            centers
                .into_iter()
                .zip(distances)
                .map(|(center, distance)| WithinDistance::new(center, distance))
                .collect()
        })
    }

    #[test]
    fn random_look_up() {
        TestRunner::default()
            .run(
                &(random_objects(100), random_queries(10)),
                |(objects, queries)| {
                    let index = KdTree::new(objects);

                    for query in queries {
                        let mut results1 = index
                            .iter()
                            .filter(|object| query.test(object.position()))
                            .collect::<Vec<_>>();

                        let mut results2 = Vec::new();
                        index.look_up(&query, |object| {
                            results2.push(object);
                            ControlFlow::Continue(())
                        });

                        results1.sort_unstable();
                        results2.sort_unstable();
                        assert_eq!(results1, results2);
                    }

                    Ok(())
                },
            )
            .unwrap();
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn random_par_look_up() {
        TestRunner::default()
            .run(
                &(random_objects(100), random_queries(10)),
                |(objects, queries)| {
                    let index = KdTree::par_new(objects);

                    for query in queries {
                        let mut results1 = index
                            .iter()
                            .filter(|object| query.test(object.position()))
                            .collect::<Vec<_>>();

                        let results2 = Mutex::new(Vec::new());
                        index.par_look_up(&query, |object| {
                            results2.lock().unwrap().push(object);
                        });
                        let mut results2 = results2.into_inner().unwrap();

                        results1.sort_unstable();
                        results2.sort_unstable();
                        assert_eq!(results1, results2);
                    }

                    Ok(())
                },
            )
            .unwrap();
    }
}
