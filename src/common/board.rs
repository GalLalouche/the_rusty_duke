use std::mem;

use crate::common::coordinates::Coordinates;

#[derive(Debug, Clone)]
pub struct Board<A> {
    // Row-first, i.e., every vector is a row, Board size is height, each vector has size of width.
    board: Vec<Vec<Option<A>>>,
    pub width: u16,
    pub height: u16,
}

impl<A> Board<A> {
    pub fn square(side: u16) -> Board<A> {
        let mut board = Vec::with_capacity(side.into());
        for _ in 0..side {
            let mut col = Vec::with_capacity(side.into());
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
    fn verify_bounds(&self, c: &Coordinates) -> () {
        assert!(self.is_in_bounds(c), "Coordinate {:?} is out of bounds", c)
    }
    fn place(&mut self, c: &Coordinates, a: Option<A>) -> Option<A> {
        self.verify_bounds(&c);
        let column = self.board.get_mut(usize::from(c.y)).unwrap();
        mem::replace(&mut column[usize::from(c.x)], a)
    }
    pub fn put(&mut self, c: &Coordinates, a: A) -> Option<A> {
        self.place(&c, Some(a))
    }
    pub fn get(&self, c: &Coordinates) -> Option<&A> {
        self.verify_bounds(c);
        self.board[usize::from(c.y)][usize::from(c.x)].as_ref()
    }
    pub fn get_mut(&mut self, c: &Coordinates) -> Option<&mut A> {
        self.verify_bounds(c);
        self.board.get_mut(usize::from(c.y)).and_then(|b| b.get_mut(usize::from(c.x))).unwrap().as_mut()
    }
    pub fn remove(&mut self, c: &Coordinates) -> Option<A> {
        self.place(c, None)
    }
    pub fn mv(&mut self, src: &Coordinates, dst: &Coordinates) -> Option<A> {
        let e: Option<A> = self.remove(src);
        assert!(e.is_some(), "Cannot move unoccupied coordinates {:?} in board", src);
        let result = self.remove(dst);
        self.place(dst, e);
        result
    }
    pub fn is_in_bounds(&self, c: &Coordinates) -> bool {
        c.x < self.width && c.y < self.height
    }
    pub fn is_out_of_bounds(&self, c: &Coordinates) -> bool {
        !self.is_in_bounds(c)
    }
    pub fn is_occupied(&self, c: &Coordinates) -> bool {
        self.get(c).is_some()
    }
    pub fn is_empty(&self, c: &Coordinates) -> bool {
        self.get(&c).is_none()
    }

    pub fn rows(&self) -> &Vec<Vec<Option<A>>> {
        &self.board
    }

    pub fn coordinates(&self) -> Vec<Coordinates> {
        (0..self.width)
            .flat_map(move |x| (0..self.height).map(move |y| Coordinates { x, y }))
            .collect()
    }

    pub fn all_coordinated_values(&self) -> Vec<(Coordinates, Option<&A>)> {
        self.coordinates().iter().map(|c| (*c, self.get(c))).collect()
    }
    pub fn active_coordinates(&self) -> Vec<(Coordinates, &A)> {
        self.coordinates()
            .iter()
            .filter_map(|c| self.get(c).map(|e| (*c, e)))
            .collect()
    }

    pub fn find<P>(&self, predicate: P) -> Option<Coordinates> where P: Fn(&A) -> bool {
        self.active_coordinates().iter().find(|(_, a)| predicate(a)).map(|(c, _)| c).cloned()
    }
}

impl<A> Board<A> where A: Clone {
    pub fn flip_vertical(&self) -> Board<A> {
        let mut res = Vec::with_capacity(self.height.into());
        for _ in 0..self.height {}
        for v in self.board.iter().rev() {
            res.push(v.clone())
        }
        Board { board: res, width: self.width, height: self.height }
    }
}

#[cfg(test)]
mod test {
    use crate::{assert_none, assert_some};

    use super::*;

    #[test]
    fn get_indexing() {
        let mut board = Board::square(2);
        board.put(&Coordinates { x: 1, y: 0 }, 1);
        assert_some!(
            1,
            board.get(&Coordinates{x: 1, y: 0}).cloned()
        );
        assert_none!(
            board.get(&Coordinates{x: 0, y: 1})
        );
    }

    #[test]
    fn get_mut_indexing() {
        let mut board = Board::square(2);
        board.put(&Coordinates { x: 1, y: 0 }, 1);
        let c = board.get_mut(&Coordinates { x: 1, y: 0 }).unwrap();
        *c += 1;
        assert_some!(
            2,
            board.get(&Coordinates { x: 1, y: 0 }).cloned()
        );
        assert_none!(
            board.get_mut(&Coordinates{x: 0, y: 1})
        );
    }

    #[test]
    fn rows_returns_the_rows_no_columns() {
        let mut board = Board::square(2);
        board.put(&Coordinates { x: 0, y: 0 }, 0);
        board.put(&Coordinates { x: 1, y: 0 }, 1);
        board.put(&Coordinates { x: 1, y: 1 }, 3);
        assert_eq!(
            &vec![vec![Some(0), Some(1)], vec![None, Some(3)]],
            board.rows(),
        )
    }

    #[test]
    fn find_returns_the_correct_coordinates_if_it_exists() {
        let mut board = Board::square(2);
        board.put(&Coordinates { x: 0, y: 0 }, 0);
        board.put(&Coordinates { x: 1, y: 0 }, 1);
        board.put(&Coordinates { x: 1, y: 1 }, 3);
        assert_some!(
            Coordinates { x: 1, y: 1 },
            board.find(|a| *a > 2),
        )
    }

    #[test]
    fn find_returns_none_if_it_doesnt_exists() {
        let mut board = Board::square(2);
        board.put(&Coordinates { x: 0, y: 0 }, 0);
        board.put(&Coordinates { x: 1, y: 0 }, 1);
        board.put(&Coordinates { x: 1, y: 1 }, 3);
        assert_none!(board.find(|a| *a < 0))
    }

    #[test]
    #[should_panic]
    fn mv_should_panic_on_empty() {
        let mut board = Board::square(2);
        board.put(&Coordinates { x: 0, y: 0 }, 0);
        board.put(&Coordinates { x: 1, y: 0 }, 1);
        board.put(&Coordinates { x: 1, y: 1 }, 3);
        board.mv(&Coordinates { x: 0, y: 1 }, &Coordinates { x: 1, y: 1 });
    }

    #[test]
    #[should_panic]
    fn mv_should_move_to_unoccupied() {
        let mut board = Board::square(2);
        board.put(&Coordinates { x: 0, y: 0 }, 0);
        board.put(&Coordinates { x: 1, y: 0 }, 1);
        board.put(&Coordinates { x: 1, y: 1 }, 3);

        let result = board.mv(&Coordinates { x: 1, y: 1 }, &Coordinates { x: 0, y: 1 });

        assert_none!(result);
        assert_some!(
            3,
            board.get(&Coordinates { x: 1, y: 0 }).cloned(),
        );
        assert_none!(board.get(&Coordinates { x: 1, y: 1 }));
    }

    #[test]
    fn mv_should_move_to_occupied() {
        let mut board = Board::square(2);
        board.put(&Coordinates { x: 0, y: 0 }, 0);
        board.put(&Coordinates { x: 1, y: 0 }, 1);
        board.put(&Coordinates { x: 1, y: 1 }, 3);

        let result = board.mv(&Coordinates { x: 1, y: 1 }, &Coordinates { x: 0, y: 0 });

        assert_some!(
            0,
            result,
        );
        assert_some!(
            3,
            board.get(&Coordinates { x: 0, y: 0 }).cloned(),
        );
        assert_none!(board.get(&Coordinates { x: 1, y: 1 }));
    }

    #[test]
    fn flip_vertical_should_flip_vertical() {
        let mut board = Board::square(2);
        board.put(&Coordinates { x: 0, y: 0 }, 0);
        board.put(&Coordinates { x: 1, y: 0 }, 1);
        board.put(&Coordinates { x: 1, y: 1 }, 3);

        let mut expected = Board::square(2);
        expected.put(&Coordinates { x: 0, y: 1 }, 0);
        expected.put(&Coordinates { x: 1, y: 1 }, 1);
        expected.put(&Coordinates { x: 1, y: 0 }, 3);

        assert_eq!(
            expected.board,
            board.flip_vertical().board,
        )
    }
}