#[cfg(feature = "rayon")]
use rayon::join;

use crate::{KdTree, Object, Point};

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
