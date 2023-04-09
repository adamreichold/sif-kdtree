use std::mem::swap;

use crate::{split, KdTree, Object, Point};

impl<O: Object> KdTree<O> {
    /// TODO
    pub fn nearest(&self, target: &O::Point) -> Option<&O> {
        let mut args = NearestArgs {
            target,
            distance_2: f64::INFINITY,
            best_match: None,
        };

        if !self.objects.is_empty() {
            nearest(&mut args, &self.objects, 0);
        }

        args.best_match
    }
}

struct NearestArgs<'a, 'b, O: Object> {
    target: &'b O::Point,
    distance_2: f64,
    best_match: Option<&'a O>,
}

fn nearest<'a, O: Object>(
    args: &mut NearestArgs<'a, '_, O>,
    mut objects: &'a [O],
    mut axis: usize,
) {
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

        let next_axis = (axis + 1) % O::Point::DIM;

        if !left.is_empty() {
            nearest(args, left, next_axis);
        }

        if right.is_empty() || args.distance_2 <= offset.powi(2) {
            return;
        }

        objects = right;
        axis = next_axis;
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
