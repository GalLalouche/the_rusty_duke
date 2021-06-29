extern crate derive_more;

use std::mem;

use derive_more::Display;
#[macro_use]
use fstrings::*;

use crate::common::utils::panic_if;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Display)]
#[display(fmt = "Coordinate({}, {})", x, y)]
pub struct Coordinates {
    pub x: usize,
    pub y: usize,
}

impl Coordinates {
    pub fn to_int_coordinate(&self) -> IntCoordinates {
        IntCoordinates { x: self.x as i16, y: self.y as i16 }
    }
}

/// Useful when substracting.
struct IntCoordinates {
    pub x: i16,
    pub y: i16,
}

impl Coordinates {
    /// Panics if src isn't on a linear (horizontal, vertical, or bishop-like diagonal to dst, 
    /// or if src == dst.
    pub fn linear_path_to(&self, dst: Coordinates) -> Vec<Coordinates> {
        panic_if(self == &dst, &f!("Can't take linear path from {dst} to itself"));
        let int_self = self.to_int_coordinate();
        let int_dst = dst.to_int_coordinate();
        // TODO use macros to avoid this ugly ass duplication
        if self.x == dst.x {
            return if self.y < dst.y {
                (self.y + 1..dst.y).map(|y| Coordinates { x: self.x, y }).collect()
            } else {
                (dst.y + 1..self.y).rev().map(|y| Coordinates { x: self.x, y }).collect()
            };
        }
        if self.y == dst.y {
            return if self.x < dst.x {
                (self.x + 1..dst.x).map(|x| Coordinates { x, y: self.y }).collect()
            } else {
                (dst.x + 1..self.x).rev().map(|x| Coordinates { x, y: self.y }).collect()
            };
        }
        // TODO reverse diagonal
        if ((int_self.x - int_dst.x) == (int_self.y - int_dst.y) {
            return if self.x < dst.x {
                (1..(int_self.x - int_dst.x).abs() as usize)
                    .map(|i| Coordinates { x: self.x + i, y: self.y + i })
                    .collect()
            } else {
                println!("{:?}", (1..(int_self.x - int_dst.x).abs() as usize).rev().collect::<Vec<_>>());

                (1..(int_self.x - int_dst.x).abs() as usize)
                    .rev()
                    .map(|i| Coordinates { x: dst.x + i, y: dst.y + i })
                    .collect()
            };
        };
        panic!("{} isn't linear to {}", self, dst);
    }
}

#[derive(Clone)]
pub struct Board<A> {
    board: Vec<Vec<Option<A>>>,
    pub width: usize,
    pub height: usize,
}

impl<A> Board<A> {
    pub fn square(side: usize) -> Board<A> {
        let mut board = Vec::with_capacity(side);
        for _ in 0..side {
            let mut col = Vec::with_capacity(side);
            for _ in 0..side {
                col.push(None);
            }
            board.push(col);
        }

        Board {
            width: side,
            height: side,
            board,
        }
    }
    fn verify_bounds(&self, c: Coordinates) -> () {
        if self.is_out_of_bounds(c) {
            panic!("Coordinate {} is out of bounds", c)
        }
    }
    fn place(&mut self, c: Coordinates, a: Option<A>) -> Option<A> {
        self.verify_bounds(c);
        let column = self.board.get_mut(c.x).unwrap();
        mem::replace(&mut column[c.y], a)
    }
    pub fn put(&mut self, c: Coordinates, a: A) -> Option<A> {
        self.place(c, Some(a))
    }
    pub fn get(&self, c: Coordinates) -> Option<&A> {
        self.verify_bounds(c);
        self.board[c.x][c.y].as_ref()
    }
    pub fn get_mut(&mut self, c: Coordinates) -> Option<&mut A> {
        self.verify_bounds(c);
        self.board.get_mut(c.x).and_then(|b| b.get_mut(c.y)).unwrap().as_mut()
    }
    pub fn remove(&mut self, c: Coordinates) -> Option<A> {
        self.place(c, None)
    }
    pub fn is_in_bounds(&self, c: Coordinates) -> bool {
        c.x < self.width && c.y < self.height
    }
    pub fn is_out_of_bounds(&self, c: Coordinates) -> bool {
        !self.is_in_bounds(c)
    }
    pub fn is_occupied(&self, c: Coordinates) -> bool {
        self.get(c).is_some()
    }
    pub fn is_empty(&self, c: Coordinates) -> bool {
        self.get(c).is_none()
    }

    pub fn as_matrix(&self) -> &Vec<Vec<Option<A>>> {
        &self.board
    }
    pub fn coordinates(&self) -> Vec<Coordinates> {
        (0..self.width)
            .flat_map(move |x| (0..self.height).map(move |y| Coordinates { x, y }))
            .collect()
    }
    pub fn active_coordinates(&self) -> Vec<(Coordinates, &A)> {
        self.coordinates()
            .iter()
            .filter_map(|c| self.get(*c).map(|e| (c.clone(), e)))
            .collect()
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
    fn linear_path_to_main_diagonal_positive() {
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
    fn linear_path_to_main_diagonal_negative() {
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
            Coordinates { x: 7, y: 6 }.linear_path_to(Coordinates { x: 3, y: 2 })
        )
    }
}