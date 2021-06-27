extern crate derive_more;

use std::mem;

use derive_more::Display;

#[derive(PartialEq, Eq, Clone, Copy, Display)]
#[display(fmt = "Coordinate({}, {})", x, y)]
pub struct Coordinate {
    pub x: usize,
    pub y: usize,
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
        for i in 0..side {
            board[i] = Vec::with_capacity(side);
        }

        Board {
            width: side,
            height: side,
            board,
        }
    }
    fn verify_bounds(&self, c: Coordinate) -> () {
        if self.is_out_of_bounds(c) {
            panic!("Coordinate {} is out of bounds", c)
        }
    }
    fn place(&mut self, c: Coordinate, a: Option<A>) -> Option<A> {
        self.verify_bounds(c);
        let column = self.board.get_mut(c.x).unwrap();
        mem::replace(&mut column[c.y], a)
    }
    pub fn put(&mut self, c: Coordinate, a: A) -> Option<A> {
        self.place(c, Some(a))
    }
    pub fn get(&self, c: Coordinate) -> Option<&A> {
        self.verify_bounds(c);
        self.board[c.x][c.y].as_ref()
    }
    pub fn get_mut(&mut self, c: Coordinate) -> Option<&mut A> {
        self.verify_bounds(c);
        self.board.get_mut(c.x).and_then(|b| b.get_mut(c.y)).unwrap().as_mut()
    }
    pub fn remove(&mut self, c: Coordinate) -> Option<A> {
        self.place(c, None)
    }
    pub fn is_in_bounds(&self, c: Coordinate) -> bool {
        c.x < self.width && c.y < self.height
    }
    pub fn is_out_of_bounds(&self, c: Coordinate) -> bool {
        !self.is_in_bounds(c)
    }
    pub fn is_occupied(&self, c: Coordinate) -> bool {
        self.get(c).is_some()
    }
    pub fn is_empty(&self, c: Coordinate) -> bool {
        self.get(c).is_none()
    }

    pub fn vecs(&self) -> &Vec<Vec<Option<A>>> {
        &self.board
    }
    pub fn coordinates(&self) -> Vec<Coordinate> {
        (0..self.width)
            .flat_map(move |x| (0..self.height).map(move |y| Coordinate { x, y }))
            .collect()
    }
    pub fn active_coordinates(&self) -> Vec<(Coordinate, &A)> {
        self.coordinates()
            .iter()
            .filter_map(|c| self.get(*c).map(|e| (c.clone(), e)))
            .collect()
    }
}