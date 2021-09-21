use fstrings::*;

use crate::common::utils::Distance;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Coordinates {
    pub x: u16,
    pub y: u16,
}

impl Coordinates {
    /// `panic`s if src isn't on a linear (horizontal, vertical, or bishop-like diagonal to `dst`,
    /// or if `src == dst`.
    pub fn linear_path_to(self, dst: Coordinates) -> Vec<Coordinates> {
        assert_ne!(self, dst, "{}", f!("Can't take linear path from {dst:?} to itself"));
        // TODO use macros to avoid this ugly ass duplication
        if self.x == dst.x {
            macro_rules! collect_y {
                ($i: expr) => {
                    $i.map(|y| Coordinates {x: self.x, y}).collect()
                }
            }
            return if self.y < dst.y {
                collect_y!(self.y + 1..dst.y)
            } else {
                collect_y!((dst.y + 1..self.y).rev())
            };
        }
        if self.y == dst.y {
            macro_rules! collect_x {
                ($i: expr) => {
                    $i.map(|x| Coordinates {x, y: self.y}).collect()
                }
            }
            return if self.x < dst.x {
                collect_x!(self.x + 1..dst.x)
            } else {
                collect_x!((dst.x + 1..self.x).rev())
            };
        }
        if self.x.distance_to(dst.x) == self.y.distance_to(dst.y) {
            macro_rules! collect {
                ($m: tt, $c: expr, $x_op: tt, $y_op: tt) => {
                     (1..self.x.distance_to(dst.x)).$m()
                        .map(|i| Coordinates { x: $c.x $x_op i, y: $c.y $y_op i})
                        .collect()
                }
            }
            return match (self.x < dst.x, self.y < dst.y) {
                (true, true) => collect!(fuse, self, +, +),
                (false, false) => collect!(rev, dst, +, +),
                (true, false) => collect!(fuse, self, +, -),
                (false, true) => collect!(rev, dst, +, -),
            };
        }
        panic!("{:?} isn't linear to {:?}", self, dst);
    }

    pub fn is_linear_to(self, dst: Coordinates) -> bool {
        self.x == dst.x ||
            self.y == dst.y ||
            self.x.distance_to(dst.x) == self.y.distance_to(dst.y)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn linear_path_to_horizontal_positive() {
        assert_eq!(
            vec![
                Coordinates { x: 1, y: 3 },
                Coordinates { x: 2, y: 3 },
            ],
            Coordinates { x: 0, y: 3 }.linear_path_to(Coordinates { x: 3, y: 3 })
        )
    }

    #[test]
    fn linear_path_to_horizontal_negative() {
        assert_eq!(
            vec![
                Coordinates { x: 2, y: 3 },
                Coordinates { x: 1, y: 3 },
            ],
            Coordinates { x: 3, y: 3 }.linear_path_to(Coordinates { x: 0, y: 3 })
        )
    }

    #[test]
    fn linear_path_to_vertical_positive() {
        assert_eq!(
            vec![
                Coordinates { x: 3, y: 1 },
                Coordinates { x: 3, y: 2 },
            ],
            Coordinates { x: 3, y: 0 }.linear_path_to(Coordinates { x: 3, y: 3 })
        )
    }

    #[test]
    fn linear_path_to_vertical_negative() {
        assert_eq!(
            vec![
                Coordinates { x: 3, y: 2 },
                Coordinates { x: 3, y: 1 },
            ],
            Coordinates { x: 3, y: 3 }.linear_path_to(Coordinates { x: 3, y: 0 })
        )
    }

    #[test]
    fn linear_path_to_main_diagonal_positive_x() {
        assert_eq!(
            vec![
                Coordinates { x: 4, y: 3 },
                Coordinates { x: 5, y: 4 },
                Coordinates { x: 6, y: 5 },
            ],
            Coordinates { x: 3, y: 2 }.linear_path_to(Coordinates { x: 7, y: 6 })
        )
    }

    #[test]
    fn linear_path_to_main_diagonal_negative_x() {
        assert_eq!(
            vec![
                Coordinates { x: 6, y: 5 },
                Coordinates { x: 5, y: 4 },
                Coordinates { x: 4, y: 3 },
            ],
            Coordinates { x: 7, y: 6 }.linear_path_to(Coordinates { x: 3, y: 2 })
        )
    }

    #[test]
    fn linear_path_to_reverse_diagonal_positive_x() {
        assert_eq!(
            vec![
                Coordinates { x: 4, y: 5 },
                Coordinates { x: 5, y: 4 },
                Coordinates { x: 6, y: 3 },
            ],
            Coordinates { x: 3, y: 6 }.linear_path_to(Coordinates { x: 7, y: 2 })
        )
    }

    #[test]
    fn linear_path_to_reverse_diagonal_negative_x() {
        assert_eq!(
            vec![
                Coordinates { x: 6, y: 3 },
                Coordinates { x: 5, y: 4 },
                Coordinates { x: 4, y: 5 },
            ],
            Coordinates { x: 7, y: 2 }.linear_path_to(Coordinates { x: 3, y: 6 })
        )
    }
}
