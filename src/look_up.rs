use std::ops::ControlFlow;

use num_traits::Num;
#[cfg(feature = "rayon")]
use rayon::join;

use crate::{contains, split, Distance, KdTree, Object, Point};

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

/// A query which yields all objects within a given axis-aligned boundary box (AABB) in `N`-dimensional space
#[derive(Debug)]
pub struct WithinBoundingBox<T, const N: usize> {
    aabb: ([T; N], [T; N]),
}

impl<T, const N: usize> WithinBoundingBox<T, N> {
    /// Construct a query from first the corner smallest coordinate values `lower` and then the corner with the largest coordinate values `upper`
    pub fn new(lower: [T; N], upper: [T; N]) -> Self {
        Self {
            aabb: (lower, upper),
        }
    }
}

impl<T, const N: usize> Query<[T; N]> for WithinBoundingBox<T, N>
where
    T: Num + Copy + PartialOrd,
{
    fn aabb(&self) -> &([T; N], [T; N]) {
        &self.aabb
    }

    fn test(&self, _position: &[T; N]) -> bool {
        true
    }
}

/// A query which yields all objects within a given distance to a central point in `N`-dimensional real space
#[derive(Debug)]
pub struct WithinDistance<T, const N: usize> {
    aabb: ([T; N], [T; N]),
    center: [T; N],
    distance_2: T,
}

impl<T, const N: usize> WithinDistance<T, N>
where
    T: Num + Copy + PartialOrd,
{
    /// Construct a query from the `center` and the largest allowed Euclidean `distance` to it
    pub fn new(center: [T; N], distance: T) -> Self {
        Self {
            aabb: (
                center.map(|coord| coord - distance),
                center.map(|coord| coord + distance),
            ),
            center,
            distance_2: distance * distance,
        }
    }
}

impl<T, const N: usize> Query<[T; N]> for WithinDistance<T, N>
where
    T: Num + Copy + PartialOrd,
{
    fn aabb(&self) -> &([T; N], [T; N]) {
        &self.aabb
    }

    fn test(&self, position: &[T; N]) -> bool {
        self.center.distance_2(position) <= self.distance_2
    }
}

impl<O, S> KdTree<O, S>
where
    O: Object,
    S: AsRef<[O]>,
{
    /// Find objects matching the given `query`
    ///
    /// Queries are defined by passing an implementor of the [`Query`] trait.
    ///
    /// Objects matching the `query` are passed to the `visitor` as they are found.
    /// Depending on its [return value][`ControlFlow`], the search is continued or stopped.
    pub fn look_up<'a, Q, V, R>(&'a self, query: &Q, visitor: V) -> ControlFlow<R>
    where
        Q: Query<O::Point>,
        V: FnMut(&'a O) -> ControlFlow<R>,
    {
        let objects = self.objects.as_ref();

        if !objects.is_empty() {
            look_up(&mut LookUpArgs { query, visitor }, objects, 0)?;
        }

        ControlFlow::Continue(())
    }

    #[cfg(feature = "rayon")]
    /// Find objects matching the given `query`, in parallel
    ///
    /// Queries are defined by passing an implementor of the [`Query`] trait.
    ///
    /// Objects matching the `query` are passed to the `visitor` as they are found.
    /// In contrast to the [serial version][Self::look_up], parts of the search can continue
    /// even after it has been stopped.
    ///
    /// Requires the `rayon` feature and dispatches tasks into the current [thread pool][rayon::ThreadPool].
    pub fn par_look_up<'a, Q, V, R>(&'a self, query: &Q, visitor: V) -> ControlFlow<R>
    where
        O: Send + Sync,
        O::Point: Sync,
        Q: Query<O::Point> + Sync,
        V: Fn(&'a O) -> ControlFlow<R> + Sync,
        R: Send,
    {
        let objects = self.objects.as_ref();

        if !objects.is_empty() {
            par_look_up(&LookUpArgs { query, visitor }, objects, 0)?;
        }

        ControlFlow::Continue(())
    }
}

struct LookUpArgs<'a, Q, V> {
    query: &'a Q,
    visitor: V,
}

fn look_up<'a, O, Q, V, R>(
    args: &mut LookUpArgs<Q, V>,
    mut objects: &'a [O],
    mut axis: usize,
) -> ControlFlow<R>
where
    O: Object,
    Q: Query<O::Point>,
    V: FnMut(&'a O) -> ControlFlow<R>,
{
    loop {
        let (left, object, right) = split(objects);

        let position = object.position();

        if contains(args.query.aabb(), position) && args.query.test(position) {
            (args.visitor)(object)?;
        }

        let search_left =
            !left.is_empty() && args.query.aabb().0.coord(axis) <= position.coord(axis);

        let search_right =
            !right.is_empty() && position.coord(axis) <= args.query.aabb().1.coord(axis);

        axis = (axis + 1) % O::Point::DIM;

        match (search_left, search_right) {
            (true, true) => {
                look_up(args, left, axis)?;

                objects = right;
            }
            (true, false) => objects = left,
            (false, true) => objects = right,
            (false, false) => return ControlFlow::Continue(()),
        }
    }
}

#[cfg(feature = "rayon")]
fn par_look_up<'a, O, Q, V, R>(
    args: &LookUpArgs<Q, V>,
    mut objects: &'a [O],
    mut axis: usize,
) -> ControlFlow<R>
where
    O: Object + Send + Sync,
    O::Point: Sync,
    Q: Query<O::Point> + Sync,
    V: Fn(&'a O) -> ControlFlow<R> + Sync,
    R: Send,
{
    loop {
        let (left, object, right) = split(objects);

        let position = object.position();

        if contains(args.query.aabb(), position) && args.query.test(position) {
            (args.visitor)(object)?;
        }

        let search_left =
            !left.is_empty() && args.query.aabb().0.coord(axis) <= position.coord(axis);

        let search_right =
            !right.is_empty() && position.coord(axis) <= args.query.aabb().1.coord(axis);

        axis = (axis + 1) % O::Point::DIM;

        match (search_left, search_right) {
            (true, true) => {
                let (left, right) = join(
                    || par_look_up(args, left, axis),
                    || par_look_up(args, right, axis),
                );

                left?;
                right?;

                return ControlFlow::Continue(());
            }
            (true, false) => objects = left,
            (false, true) => objects = right,
            (false, false) => return ControlFlow::Continue(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rayon")]
    use std::sync::Mutex;

    use proptest::{collection::vec, strategy::Strategy, test_runner::TestRunner};

    use crate::tests::{random_objects, random_points};

    pub fn random_queries(len: usize) -> impl Strategy<Value = Vec<WithinDistance<f32, 2>>> {
        (random_points(len), vec(0.0_f32..=1.0, len)).prop_map(|(centers, distances)| {
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
                        index
                            .look_up(&query, |object| {
                                results2.push(object);
                                ControlFlow::<()>::Continue(())
                            })
                            .continue_value()
                            .unwrap();

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
                        index
                            .par_look_up(&query, |object| {
                                results2.lock().unwrap().push(object);
                                ControlFlow::<()>::Continue(())
                            })
                            .continue_value()
                            .unwrap();
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
