use std::marker::PhantomData;

#[cfg(feature = "rayon")]
use rayon::join;

use crate::{KdTree, Object, Point};

impl<O, S> KdTree<O, S>
where
    O: Object,
    S: AsRef<[O]> + AsMut<[O]>,
{
    /// Construct a new tree by sorting the given `objects`
    pub fn new(mut objects: S) -> Self {
        sort(objects.as_mut(), 0);

        Self {
            objects,
            _marker: PhantomData,
        }
    }

    #[cfg(feature = "rayon")]
    /// Construct a new tree by sorting the given `objects`, in parallel
    ///
    /// Requires the `rayon` feature and dispatches tasks into the current [thread pool][rayon::ThreadPool].
    pub fn par_new(mut objects: S) -> Self
    where
        O: Send,
    {
        par_sort(objects.as_mut(), 0);

        Self {
            objects,
            _marker: PhantomData,
        }
    }
}

fn sort<O>(objects: &mut [O], axis: usize)
where
    O: Object,
{
    if objects.len() <= 1 {
        return;
    }

    let (left, right, next_axis) = sort_axis(objects, axis);

    sort(left, next_axis);
    sort(right, next_axis);
}

#[cfg(feature = "rayon")]
fn par_sort<O>(objects: &mut [O], axis: usize)
where
    O: Object + Send,
{
    if objects.len() <= 1 {
        return;
    }

    let (left, right, next_axis) = sort_axis(objects, axis);

    join(|| par_sort(left, next_axis), || par_sort(right, next_axis));
}

fn sort_axis<O>(objects: &mut [O], axis: usize) -> (&mut [O], &mut [O], usize)
where
    O: Object,
{
    let mid = objects.len() / 2;

    let (left, _, right) = objects.select_nth_unstable_by(mid, |lhs, rhs| {
        let lhs = lhs.position().coord(axis);
        let rhs = rhs.position().coord(axis);

        lhs.partial_cmp(&rhs).unwrap()
    });

    let next_axis = (axis + 1) % O::Point::DIM;

    (left, right, next_axis)
}
