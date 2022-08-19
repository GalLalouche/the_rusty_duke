use std::hash::{Hash, Hasher};
use std::mem;

use crate::common::coordinates::Coordinates;
use crate::common::geometry::{Rectangular, Square};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board<A> {
    // Row-first, i.e.,
    // [1 2 3
    //  4 5 6]
    // Is represented as [1 2 3 4 5 6]
    board: Vec<Option<A>>,
    width: u16,
    height: u16,
}

impl<A> Board<A> {
    pub fn rect(r: impl Rectangular) -> Board<A> {
        let mut board = Vec::with_capacity(r.area() as usize);
        board.resize_with(r.area() as usize, || None);
        Board {
            width: r.width(),
            height: r.height(),
            board,
        }
    }
    pub fn square(side: u16) -> Board<A> { Board::rect(Square::new(side)) }
    #[inline(always)]
    fn verify_bounds(&self, c: Coordinates) -> () {
        debug_assert!(self.is_in_bounds(c), "Coordinate {:?} is out of bounds", c)
    }
    #[inline(always)]
    fn to_vec_index(&self, c: Coordinates) -> usize { (self.width * c.y + c.x) as usize }
    fn place(&mut self, c: Coordinates, a: Option<A>) -> Option<A> {
        self.verify_bounds(c);
        let index = self.to_vec_index(c);
        mem::replace(&mut self.board[index], a)
    }
    pub fn put(&mut self, c: Coordinates, a: A) -> Option<A> {
        self.place(c, Some(a))
    }
    #[inline(always)]
    pub fn get(&self, c: Coordinates) -> Option<&A> {
        self.verify_bounds(c);
        self.board[self.to_vec_index(c)].as_ref()
    }
    pub fn get_mut(&mut self, c: Coordinates) -> Option<&mut A> {
        self.verify_bounds(c);
        let index = self.to_vec_index(c);
        self.board.get_mut(index).unwrap().as_mut()
    }
    pub fn remove(&mut self, c: Coordinates) -> Option<A> {
        self.place(c, None)
    }
    pub fn mv(&mut self, src: Coordinates, dst: Coordinates) -> Option<A> {
        let e: Option<A> = self.remove(src);
        assert!(e.is_some(), "Cannot move unoccupied coordinates {:?} in board", src);
        let result = self.remove(dst);
        self.place(dst, e);
        result
    }
    pub fn is_occupied(&self, c: Coordinates) -> bool {
        self.get(c).is_some()
    }
    pub fn is_empty(&self, c: Coordinates) -> bool {
        self.get(c).is_none()
    }

    fn coordinates(&self) -> impl Iterator<Item=Coordinates> + '_ {
        (0..self.width).flat_map(move |x| (0..self.height).map(move |y| Coordinates { x, y }))
    }

    pub fn all_coordinated_values(&self) -> Vec<(Coordinates, Option<&A>)> {
        self.coordinates().into_iter().map(|c| (c, self.get(c))).collect()
    }
    pub fn active_coordinates(&self) -> impl Iterator<Item=(Coordinates, &A)> + '_ {
        self.coordinates()
            .into_iter()
            .filter_map(move |c| self.get(c).map(|e| (c, e)))
    }

    pub fn find<P>(&self, predicate: P) -> Option<Coordinates> where P: Fn(&A) -> bool {
        self.active_coordinates().find(|(_, a)| predicate(a)).map(|(c, _)| c)
    }
}

impl <A: Hash> Hash for Board<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.board.hash(state)
    }
}

impl<A: Clone> Board<A> {
    pub fn flip_vertical(&self) -> Board<A> {
        let mut res = Vec::with_capacity(self.area() as usize);
        for y in (0..self.height).rev() {
            for x in 0..self.width {
                res.push(self.get(Coordinates { x, y }).cloned())
            }
        }
        Board { board: res, width: self.width, height: self.height }
    }

    pub fn rows(&self) -> Vec<Vec<Option<A>>> {
        let mut result = Vec::with_capacity(self.height as usize);
        for y in 0..self.height {
            let mut row = Vec::with_capacity(self.width as usize);
            for x in 0..self.width {
                let c = Coordinates { x, y };
                row.push(self.get(c).cloned())
            }
            result.push(row);
        }
        result
    }
}

impl<A> Rectangular for Board<A> {
    fn width(&self) -> u16 { self.width }
    fn height(&self) -> u16 { self.height }
}

#[cfg(test)]
mod test {
    use crate::{assert_none, assert_some};
    use crate::common::geometry::Rectangle;

    use super::*;

    fn make_board() -> Board<i32> { Board::rect(Rectangle::with_width_and_height(3, 2)) }

    #[test]
    fn get_indexing() {
        let mut board = make_board();
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
        let mut board = make_board();
        board.put(Coordinates { x: 1, y: 0 }, 1);
        let c = board.get_mut(Coordinates { x: 1, y: 0 }).unwrap();
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
    fn rows_returns_the_rows_not_columns() {
        let mut board = make_board();
        board.put(Coordinates { x: 0, y: 0 }, 0);
        board.put(Coordinates { x: 1, y: 0 }, 1);
        board.put(Coordinates { x: 2, y: 1 }, 4);
        board.put(Coordinates { x: 1, y: 1 }, 3);
        assert_eq!(
            vec![vec![Some(0), Some(1), None], vec![None, Some(3), Some(4)]],
            board.rows(),
        )
    }

    #[test]
    fn find_returns_the_correct_coordinates_if_it_exists() {
        let mut board = make_board();
        board.put(Coordinates { x: 0, y: 0 }, 0);
        board.put(Coordinates { x: 1, y: 0 }, 1);
        board.put(Coordinates { x: 1, y: 1 }, 3);
        assert_some!(
            Coordinates { x: 1, y: 1 },
            board.find(|a| *a > 2),
        )
    }

    #[test]
    fn find_returns_none_if_it_doesnt_exists() {
        let mut board = make_board();
        board.put(Coordinates { x: 0, y: 0 }, 0);
        board.put(Coordinates { x: 1, y: 0 }, 1);
        board.put(Coordinates { x: 1, y: 1 }, 3);
        assert_none!(board.find(|a| *a < 0))
    }

    #[test]
    #[should_panic]
    fn mv_should_panic_on_empty() {
        let mut board = make_board();
        board.put(Coordinates { x: 0, y: 0 }, 0);
        board.put(Coordinates { x: 1, y: 0 }, 1);
        board.put(Coordinates { x: 1, y: 1 }, 3);
        board.mv(Coordinates { x: 0, y: 1 }, Coordinates { x: 1, y: 1 });
    }

    #[test]
    #[should_panic]
    fn mv_should_move_to_unoccupied() {
        let mut board = make_board();
        board.put(Coordinates { x: 0, y: 0 }, 0);
        board.put(Coordinates { x: 1, y: 0 }, 1);
        board.put(Coordinates { x: 1, y: 1 }, 3);

        let result = board.mv(Coordinates { x: 1, y: 1 }, Coordinates { x: 0, y: 1 });

        assert_none!(result);
        assert_some!(
            3,
            board.get(Coordinates { x: 1, y: 0 }).cloned(),
        );
        assert_none!(board.get(Coordinates { x: 1, y: 1 }));
    }

    #[test]
    fn mv_should_move_to_occupied() {
        let mut board = make_board();
        board.put(Coordinates { x: 0, y: 0 }, 0);
        board.put(Coordinates { x: 1, y: 0 }, 1);
        board.put(Coordinates { x: 1, y: 1 }, 3);

        let result = board.mv(Coordinates { x: 1, y: 1 }, Coordinates { x: 0, y: 0 });

        assert_some!(
            0,
            result,
        );
        assert_some!(
            3,
            board.get(Coordinates { x: 0, y: 0 }).cloned(),
        );
        assert_none!(board.get(Coordinates { x: 1, y: 1 }));
    }

    #[test]
    fn flip_vertical_should_flip_vertical() {
        let mut board = make_board();
        board.put(Coordinates { x: 0, y: 0 }, 0);
        board.put(Coordinates { x: 1, y: 0 }, 1);
        board.put(Coordinates { x: 1, y: 1 }, 3);
        board.put(Coordinates { x: 2, y: 0 }, 4);

        let mut expected = make_board();
        expected.put(Coordinates { x: 0, y: 1 }, 0);
        expected.put(Coordinates { x: 1, y: 1 }, 1);
        expected.put(Coordinates { x: 1, y: 0 }, 3);
        expected.put(Coordinates { x: 2, y: 1 }, 4);

        assert_eq!(
            expected.board,
            board.flip_vertical().board,
        )
    }
}