use std::mem::swap;

use num_traits::Float;

use crate::{split, Distance, KdTree, Object, Point};

impl<O, S> KdTree<O, S>
where
    O: Object,
    O::Point: Distance,
    <O::Point as Point>::Coord: Float,
    S: AsRef<[O]>,
{
    /// Find the object nearest to the given `target`
    ///
    /// The distance is determined according to [`Point::distance_2`].
    ///
    /// Returns `None` if the tree is empty or if no object has a finite distance to the `target`.
    pub fn nearest(&self, target: &O::Point) -> Option<&O> {
        let mut args = NearestArgs {
            target,
            distance_2: <O::Point as Point>::Coord::infinity(),
            best_match: None,
        };

        let objects = self.objects.as_ref();

        if !objects.is_empty() {
            nearest(&mut args, objects, 0);
        }

        args.best_match
    }
}

struct NearestArgs<'a, 'b, O>
where
    O: Object,
{
    target: &'b O::Point,
    distance_2: <O::Point as Point>::Coord,
    best_match: Option<&'a O>,
}

fn nearest<'a, O>(args: &mut NearestArgs<'a, '_, O>, mut objects: &'a [O], mut axis: usize)
where
    O: Object,
    O::Point: Distance,
    <O::Point as Point>::Coord: Float,
{
    loop {
        let (mut left, object, mut right) = split(objects);

        let position = object.position();

        let distance_2 = args.target.distance_2(position);

        if args.distance_2 > distance_2 {
            args.distance_2 = distance_2;
            args.best_match = Some(object);
        }

        let offset = args.target.coord(axis) - position.coord(axis);

        if offset.is_sign_positive() {
            swap(&mut left, &mut right);
        }

        let search_left = !left.is_empty();
        let search_right = !right.is_empty();

        axis = (axis + 1) % O::Point::DIM;

        if search_right {
            if search_left {
                nearest(args, left, axis);
            }

            if args.distance_2 > offset.powi(2) {
                objects = right;
            } else {
                return;
            }
        } else if search_left {
            objects = left;
        } else {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::test_runner::TestRunner;

    use crate::tests::{random_objects, random_points};

    #[test]
    fn random_nearest() {
        TestRunner::default()
            .run(
                &(random_objects(100), random_points(10)),
                |(objects, targets)| {
                    let index = KdTree::new(objects);

                    for target in targets {
                        let result1 = index
                            .iter()
                            .min_by(|lhs, rhs| {
                                let lhs = lhs.0.distance_2(&target);
                                let rhs = rhs.0.distance_2(&target);

                                lhs.partial_cmp(&rhs).unwrap()
                            })
                            .unwrap();

                        let result2 = index.nearest(&target).unwrap();

                        assert_eq!(result1, result2);
                    }

                    Ok(())
                },
            )
            .unwrap();
    }
}
