extern crate derive_more;


use std::mem;
use crate::common::coordinates::Coordinates;

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
        assert!(self.is_in_bounds(c), "Coordinate {} is out of bounds", c)
    }
    fn place(&mut self, c: Coordinates, a: Option<A>) -> Option<A> {
        self.verify_bounds(c);
        let column = self.board.get_mut(c.y).unwrap();
        mem::replace(&mut column[c.x], a)
    }
    pub fn put(&mut self, c: Coordinates, a: A) -> Option<A> {
        self.place(c, Some(a))
    }
    pub fn get(&self, c: Coordinates) -> Option<&A> {
        self.verify_bounds(c);
        self.board[c.y][c.x].as_ref()
    }
    pub fn get_mut(&mut self, c: Coordinates) -> Option<&mut A> {
        self.verify_bounds(c);
        self.board.get_mut(c.y).and_then(|b| b.get_mut(c.x)).unwrap().as_mut()
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

    pub fn rows(&self) -> &Vec<Vec<Option<A>>> {
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
            .filter_map(|c| self.get(*c).map(|e| (*c, e)))
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{assert_some, assert_none};

    #[test]
    fn get_indexing() {
        let mut board = Board::square(2);
        board.put(Coordinates { x: 1, y: 0 }, 1);
        assert_some!(
            1,
            board.get(Coordinates{x: 1, y: 0}).cloned()
        );
        assert_none!(
            board.get(Coordinates{x: 0, y: 1})
        );
    }

    #[test]
    fn get_mut_indexing() {
        let mut board = Board::square(2);
        board.put(Coordinates { x: 1, y: 0 }, 1);
        let mut c = board.get_mut(Coordinates { x: 1, y: 0 }).unwrap();
        *c += 1;
        assert_some!(
            2,
            board.get(Coordinates { x: 1, y: 0 }).cloned()
        );
        assert_none!(
            board.get_mut(Coordinates{x: 0, y: 1})
        );
    }

    #[test]
    fn rows_returns_the_rows_no_columns() {
        let mut board = Board::square(2);
        board.put(Coordinates { x: 0, y: 0 }, 0);
        board.put(Coordinates { x: 1, y: 0 }, 1);
        board.put(Coordinates { x: 1, y: 1 }, 3);
        assert_eq!(
            &vec![vec![Some(0), Some(1)], vec![None, Some(3)]],
            board.rows(),
        )
    }
}