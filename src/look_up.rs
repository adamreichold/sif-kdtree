use std::ops::ControlFlow;

#[cfg(feature = "rayon")]
use rayon::join;

use crate::{contains, split, KdTree, Object, Point};

/// Defines a spatial query by its axis-aligned bounding box (AABB) and a method to test a single point
///
/// The AABB of the query is used to limit the points which are tested and therefore the AABB should be as tight as possible while staying aligned to the coordinate axes.
/// The test method itself can then be relatively expensive like determining the distance of the given position to an arbitrary polygon.
///
/// A very simple example of implementing this trait is [`WithinBoundingBox`] whereas a very common example is [`WithinDistance`].
pub trait Query<P: Point> {
    /// Return the axis-aligned bounding box (AABB) of the query
    ///
    /// Represented by the corners with first the smallest and then the largest coordinate values.
    ///
    /// Note that calling this method is assumed to be cheap, returning a reference to an AABB stored in the interior of the object.
    fn aabb(&self) -> &(P, P);

    /// Check whether a given `position` inside the [axis-aligned bounding box (AABB)][Self::aabb] machtes the query.
    fn test(&self, position: &P) -> bool;
}

/// A query which yields all objects within a given axis-aligned boundary box (AABB) in `N`-dimensional real space
#[derive(Debug)]
pub struct WithinBoundingBox<const N: usize> {
    aabb: ([f64; N], [f64; N]),
}

impl<const N: usize> WithinBoundingBox<N> {
    /// Construct a query from first the corner smallest coordinate values `lower` and then the corner with the largest coordinate values `upper`
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

/// A query which yields all objects within a given distance to a central point in `N`-dimensional real space
#[derive(Debug)]
pub struct WithinDistance<const N: usize> {
    aabb: ([f64; N], [f64; N]),
    center: [f64; N],
    distance_2: f64,
}

impl<const N: usize> WithinDistance<N> {
    /// Construct a query from the `center` and the largest allowed Euclidean `distance` to it
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

impl<O> KdTree<O>
where
    O: Object,
{
    /// Find objects matching the given `query`
    ///
    /// Queries are defined by passing an implementor of the [`Query`] trait.
    ///
    /// Objects matching the `query` are passed to the `visitor` as they are found.
    /// Depending on its [return value][`ControlFlow`], the search is continued or stopped.
    pub fn look_up<'a, Q, V>(&'a self, query: &Q, visitor: V)
    where
        Q: Query<O::Point>,
        V: FnMut(&'a O) -> ControlFlow<()>,
    {
        if !self.objects.is_empty() {
            look_up(&mut LookUpArgs { query, visitor }, &self.objects, 0);
        }
    }

    #[cfg(feature = "rayon")]
    /// Find objects matching the given `query`, in parallel
    ///
    /// Queries are defined by passing an implementor of the [`Query`] trait.
    ///
    /// Objects matching the `query` are passed to the `visitor` as they are found.
    /// In contrast to the [serial version][Self::look_up], the search cannot be stopped early.
    ///
    /// Requires the `rayon` feature and dispatches tasks into the current [thread pool][rayon::ThreadPool].
    pub fn par_look_up<'a, Q, V>(&'a self, query: &Q, visitor: V)
    where
        O: Send + Sync,
        O::Point: Sync,
        Q: Query<O::Point> + Sync,
        V: Fn(&'a O) + Sync,
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

fn look_up<'a, O, Q, V>(
    args: &mut LookUpArgs<Q, V>,
    mut objects: &'a [O],
    mut axis: usize,
) -> ControlFlow<()>
where
    O: Object,
    Q: Query<O::Point>,
    V: FnMut(&'a O) -> ControlFlow<()>,
{
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

        if !right.is_empty() && position.coord(axis) <= args.query.aabb().1.coord(axis) {
            objects = right;
            axis = next_axis;
        } else {
            return ControlFlow::Continue(());
        }
    }
}

#[cfg(feature = "rayon")]
fn par_look_up<'a, O, Q, V>(args: &LookUpArgs<Q, V>, objects: &'a [O], axis: usize)
where
    O: Object + Send + Sync,
    O::Point: Sync,
    Q: Query<O::Point> + Sync,
    V: Fn(&'a O) + Sync,
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
            if !right.is_empty() && position.coord(axis) <= args.query.aabb().1.coord(axis) {
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
