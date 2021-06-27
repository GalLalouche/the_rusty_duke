use crate::common::board::{Board, Coordinate};
use crate::game::token::{GameToken};

pub struct GameBoard {
    board: Board<GameToken>,
}

impl GameBoard {
    const BOARD_SIZE: usize = 6;
    pub fn height(&self) -> usize {
        self.board.height
    }
    pub fn width(&self) -> usize {
        self.board.width
    }
    pub fn empty() -> GameBoard {
        GameBoard { board: Board::square(GameBoard::BOARD_SIZE) }
    }
    pub fn place(&mut self, c: Coordinate, t: GameToken) -> () {
        if self.board.is_occupied(c) {
            panic!("Cannot insert token into occupied space {}", c)
        }
        self.board.put(c, t);
    }

    pub fn coordinates(&self) -> &Vec<Vec<Option<GameToken>>> {
        self.board.vecs()
    }
}