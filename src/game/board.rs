use std::borrow::Borrow;
use std::convert::TryFrom;

use crate::common::board::Board;
use crate::common::coordinates::Coordinates;
use crate::game::offset;
use crate::game::offset::{HorizontalOffset, VerticalOffset};
use crate::game::tile::{OwnedTile, Owner, Ownership, TileAction, TileSide};

pub struct GameBoard {
    board: Board<OwnedTile>,
}

impl GameBoard {
    const BOARD_SIZE: u16 = 6;
    pub fn height(&self) -> u16 {
        self.board.height
    }
    pub fn width(&self) -> u16 {
        self.board.width
    }
    pub fn get_board(&self) -> &Board<OwnedTile> {
        &self.board
    }
    pub fn empty() -> GameBoard {
        GameBoard { board: Board::square(GameBoard::BOARD_SIZE) }
    }
    pub fn place(&mut self, c: Coordinates, t: OwnedTile) -> () {
        assert!(self.board.is_empty(c), "Cannot insert token into occupied space {:?}", c);
        self.board.put(c, t);
    }

    pub fn rows(&self) -> &Vec<Vec<Option<OwnedTile>>> {
        self.board.rows()
    }

    pub fn get(&self, c: Coordinates) -> Option<&OwnedTile> {
        self.board.get(c)
    }

    fn to_absolute_coordinate(
        &self, src: Coordinates, offset: offset::Offsets, center: VerticalOffset,
    ) -> Option<Coordinates> {
        let x: i32 = (src.x as i32) + match offset.x {
            HorizontalOffset::FarLeft => -2,
            HorizontalOffset::Left => -1,
            HorizontalOffset::Center => 0,
            HorizontalOffset::Right => 1,
            HorizontalOffset::FarRight => 2,
        };
        fn vertical_offset(y: VerticalOffset) -> i32 {
            match y {
                VerticalOffset::FarTop => -2,
                VerticalOffset::Top => -1,
                VerticalOffset::Center => 0,
                VerticalOffset::Bottom => 1,
                VerticalOffset::FarBottom => 2,
            }
        }
        let y: i32 = (src.x as i32) + vertical_offset(offset.y) + vertical_offset(center);
        u16::try_from(x).and_then(|x| u16::try_from(y).map(|y| Coordinates { x, y }))
            .ok()
            .filter(|e| self.board.is_in_bounds(*e))
    }

    fn unobstructed(&self, src: Coordinates, dst: Coordinates) -> bool {
        src.linear_path_to(dst).iter().all(|c| self.board.is_empty(*c))
    }
    fn can_apply(
        &self,
        owner: Owner,
        tile_side: TileSide,
        src: Coordinates,
        c: offset::Offsets, a: &TileAction
    ) -> Option<Coordinates> {
        self.to_absolute_coordinate(src, c, tile_side.center_offset())
            .filter(|dst| {
                let can_move_to =
                    self.board.get(*dst).map_or(false, |c| owner.borrow().different_team(&&c.owner));
                match a {
                    TileAction::Move => self.unobstructed(src, *dst) && can_move_to,
                    TileAction::Jump => can_move_to,
                    TileAction::Unit => unimplemented!(),
                    TileAction::Slide => unimplemented!(),
                    TileAction::Command => unimplemented!(),
                    TileAction::JumpSlide => unimplemented!(),
                    TileAction::Strike => unimplemented!(),
                }
            })
    }
}