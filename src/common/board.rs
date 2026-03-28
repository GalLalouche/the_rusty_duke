use std::hash::{Hash, Hasher};
use std::mem;

use crate::common::coordinates::Coordinates;
use crate::common::geometry::{Rectangular, Square};

/// Maximum number of cells a Board can hold (6×6 game board).
const BOARD_MAX_CELLS: usize = 36;

#[derive(Debug, Clone)]
pub struct Board<A: Copy> {
    // Row-first, i.e.,
    // [1 2 3
    //  4 5 6]
    // Is represented as [1 2 3 4 5 6]
    board: [Option<A>; BOARD_MAX_CELLS],
    width: u8,
    height: u8,
}

impl<A: Copy + PartialEq> PartialEq for Board<A> {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.board == other.board
    }
}

impl<A: Copy + Eq> Eq for Board<A> {}

impl<A: Copy> Board<A> {
    pub fn rect(r: impl Rectangular) -> Board<A> {
        assert!(
            (r.width() as usize) * (r.height() as usize) <= BOARD_MAX_CELLS,
            "Board dimensions {}x{} exceed BOARD_MAX_CELLS ({})",
            r.width(), r.height(), BOARD_MAX_CELLS,
        );
        Board {
            width: r.width(),
            height: r.height(),
            board: [None; BOARD_MAX_CELLS],
        }
    }
    pub fn square(side: u8) -> Board<A> { Board::rect(Square::new(side)) }
    #[inline(always)]
    fn verify_bounds(&self, c: Coordinates) -> () {
        debug_assert!(self.is_in_bounds(c), "Coordinate {:?} is out of bounds", c)
    }
    #[inline(always)]
    fn to_vec_index(&self, c: Coordinates) -> usize { (self.width as usize) * (c.y as usize) + (c.x as usize) }
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

    pub fn flip_vertical(&self) -> Board<A> {
        let mut new_board = [None; BOARD_MAX_CELLS];
        for y in 0..self.height {
            let src_y = self.height - 1 - y;
            for x in 0..self.width {
                let src = (self.width as usize) * (src_y as usize) + (x as usize);
                let dst = (self.width as usize) * (y as usize) + (x as usize);
                new_board[dst] = self.board[src];
            }
        }
        Board { board: new_board, width: self.width, height: self.height }
    }

    pub fn rows(&self) -> Vec<Vec<Option<A>>> {
        let mut result = Vec::with_capacity(self.height as usize);
        for y in 0..self.height {
            let mut row = Vec::with_capacity(self.width as usize);
            for x in 0..self.width {
                let c = Coordinates { x, y };
                row.push(self.get(c).copied())
            }
            result.push(row);
        }
        result
    }
}

impl<A: Copy + Hash> Hash for Board<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.board.hash(state)
    }
}

impl<A: Copy> Rectangular for Board<A> {
    fn width(&self) -> u8 { self.width }
    fn height(&self) -> u8 { self.height }
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

    // ── remove tests ──────────────────────────────────────────────────
    #[test]
    fn remove_returns_the_removed_element() {
        let mut board = make_board();
        board.put(Coordinates { x: 1, y: 0 }, 42);
        let removed = board.remove(Coordinates { x: 1, y: 0 });
        assert_some!(42, removed);
        assert_none!(board.get(Coordinates { x: 1, y: 0 }));
    }

    #[test]
    fn remove_returns_none_for_empty_cell() {
        let mut board = make_board();
        let removed = board.remove(Coordinates { x: 0, y: 0 });
        assert_none!(removed);
    }

    // ── is_occupied / is_empty tests ──────────────────────────────────
    #[test]
    fn is_occupied_returns_true_for_occupied_cell() {
        let mut board = make_board();
        board.put(Coordinates { x: 0, y: 0 }, 5);
        assert!(board.is_occupied(Coordinates { x: 0, y: 0 }));
    }

    #[test]
    fn is_occupied_returns_false_for_empty_cell() {
        let board = make_board();
        assert!(!board.is_occupied(Coordinates { x: 0, y: 0 }));
    }

    #[test]
    fn is_empty_returns_true_for_empty_cell() {
        let board = make_board();
        assert!(board.is_empty(Coordinates { x: 0, y: 0 }));
    }

    #[test]
    fn is_empty_returns_false_for_occupied_cell() {
        let mut board = make_board();
        board.put(Coordinates { x: 0, y: 0 }, 5);
        assert!(!board.is_empty(Coordinates { x: 0, y: 0 }));
    }

    // ── active_coordinates tests ──────────────────────────────────────
    #[test]
    fn active_coordinates_returns_only_non_empty_positions() {
        let mut board = make_board();
        board.put(Coordinates { x: 0, y: 0 }, 10);
        board.put(Coordinates { x: 2, y: 1 }, 20);
        let active: Vec<_> = board.active_coordinates().collect();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&(Coordinates { x: 0, y: 0 }, &10)));
        assert!(active.contains(&(Coordinates { x: 2, y: 1 }, &20)));
    }

    #[test]
    fn active_coordinates_empty_board_returns_nothing() {
        let board = make_board();
        let active: Vec<_> = board.active_coordinates().collect();
        assert!(active.is_empty());
    }

    // ── put returns previous value ────────────────────────────────────
    #[test]
    fn put_on_occupied_returns_previous_value() {
        let mut board = make_board();
        board.put(Coordinates { x: 1, y: 0 }, 10);
        let prev = board.put(Coordinates { x: 1, y: 0 }, 20);
        assert_some!(10, prev);
        assert_some!(20, board.get(Coordinates { x: 1, y: 0 }).cloned());
    }

    #[test]
    fn put_on_empty_returns_none() {
        let mut board = make_board();
        let prev = board.put(Coordinates { x: 1, y: 0 }, 10);
        assert_none!(prev);
    }

    // ── flip_vertical on square board ─────────────────────────────────
    #[test]
    fn flip_vertical_on_square_board() {
        let mut board: Board<i32> = Board::square(3);
        board.put(Coordinates { x: 0, y: 0 }, 1);
        board.put(Coordinates { x: 1, y: 1 }, 5);
        board.put(Coordinates { x: 2, y: 2 }, 9);

        let flipped = board.flip_vertical();
        // y=0 -> y=2, y=1 -> y=1, y=2 -> y=0
        assert_some!(1, flipped.get(Coordinates { x: 0, y: 2 }).cloned());
        assert_some!(5, flipped.get(Coordinates { x: 1, y: 1 }).cloned());
        assert_some!(9, flipped.get(Coordinates { x: 2, y: 0 }).cloned());
        assert_none!(flipped.get(Coordinates { x: 0, y: 0 }));
    }

    #[test]
    fn flip_vertical_twice_is_identity() {
        let mut board = make_board();
        board.put(Coordinates { x: 0, y: 0 }, 1);
        board.put(Coordinates { x: 2, y: 1 }, 7);
        let double_flipped = board.flip_vertical().flip_vertical();
        assert_eq!(board.board, double_flipped.board);
    }
}