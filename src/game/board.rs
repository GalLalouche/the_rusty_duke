use std::convert::TryFrom;

use crate::common::board::{Board, Coordinates};
use crate::game::offset;
use crate::game::offset::{HorizontalOffset, VerticalOffset};
use crate::game::token::{OwnedToken, Ownership, TokenAction};

pub struct GameBoard {
    board: Board<OwnedToken>,
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
    pub fn place(&mut self, c: Coordinates, t: OwnedToken) -> () {
        assert!(self.board.is_empty(c), "Cannot insert token into occupied space {}", c);
        self.board.put(c, t);
    }

    pub fn coordinates(&self) -> &Vec<Vec<Option<OwnedToken>>> {
        self.board.as_matrix()
    }

    pub fn get(&self, c: Coordinates) -> Option<&OwnedToken> {
        self.board.get(c)
    }

    fn to_absolute_coordinate(
        &self, src: Coordinates, offset: offset::Coordinate) -> Option<Coordinates> {
        let x: i32 = (src.x as i32) + match offset.x {
            HorizontalOffset::FarLeft => -2,
            HorizontalOffset::Left => -1,
            HorizontalOffset::Center => 0,
            HorizontalOffset::Right => 1,
            HorizontalOffset::FarRight => 2,
        };
        let y: i32 = (src.x as i32) + match offset.y {
            VerticalOffset::FarTop => -2,
            VerticalOffset::Top => -1,
            VerticalOffset::Center => 0,
            VerticalOffset::Bottom => 1,
            VerticalOffset::FarBottom => 2,
        };
        usize::try_from(x).and_then(|x| usize::try_from(y).map(|y| Coordinates { x, y }))
            .ok()
            .filter(|e| self.board.is_in_bounds(*e))
    }

    fn unobstructed(&self, src: Coordinates, dst: Coordinates) -> bool {
        src.linear_path_to(dst).iter().all(|c| self.board.is_empty(*c))
    }
    fn can_apply(&self, src_token: &OwnedToken, src: Coordinates, c: offset::Coordinate, a: &TokenAction) -> Option<Coordinates> {
        self.to_absolute_coordinate(src, c)
            .filter(|dst| {
                let can_move_to = self.board.get(*dst).map_or(false, |c| src_token.different_team(c));
                match a {
                    TokenAction::Move => self.unobstructed(src, *dst) && can_move_to,
                    TokenAction::Jump => can_move_to,
                    TokenAction::Slide => unimplemented!(),
                    TokenAction::Command => unimplemented!(),
                    TokenAction::JumpSlide => unimplemented!(),
                    TokenAction::Strike => unimplemented!(),
                }
            })
    }
    // // TODO handle command
    // /// Returns None if there is no unit in src.
    // pub fn get_valid_movements(&self, src: Coordinates) -> Option<Vec<Coordinates>> {
    //     self.board.get(src).map(|t| {
    //         t.token.get_current_side()
    //             .actions()
    //             .iter()
    //             .filter_map(|e| self.can_apply(e.0, e.1))
    //             .collect()
    //     })
    // }
}