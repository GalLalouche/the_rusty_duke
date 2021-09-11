use std::borrow::Borrow;
use std::convert::TryFrom;
use std::ops::Range;

use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use crate::common::board::Board;
use crate::common::coordinates::Coordinates;
use crate::common::utils::Folding;
use crate::game::offset::{Centerable, HorizontalOffset, Offsets, VerticalOffset};
use crate::game::tile::{Owner, Ownership, PlacedTile, Tile, TileAction};
use crate::game::units;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FootmenSetup {
    // Footmen are to the sides of the Duke
    Sides,
    // One Footman is above the Duke, and one is to its player's left
    Left,
    // One Footman is above the Duke, and one is to its player's right
    Right,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DukeInitialLocation { Left, Right }

#[derive(Debug, PartialEq, Eq, Clone, Copy, EnumIter)]
pub enum DukeOffset { Top, Bottom, Left, Right }

#[derive(Debug, Clone)]
pub enum GameMove {
    PlaceNewTile(Tile, DukeOffset),
    ApplyNonCommandTileAction { src: Coordinates, dst: Coordinates },
    CommandAnotherTile { commander_src: Coordinates, unit_src: Coordinates, unit_dst: Coordinates },
}

#[derive(Debug, Clone)]
pub struct GameBoard {
    board: Board<PlacedTile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AppliedPubAction { Movement, Strike, Invalid }

impl GameBoard {
    const BOARD_SIZE: u16 = 6;
    fn absolute_duke_offset(&self, offset: DukeOffset, c: Coordinates) -> Option<Coordinates> {
        fn or_none<P>(b: bool, c: P) -> Option<Coordinates> where P: Fn() -> Coordinates {
            if b { Some(c()) } else { None }
        }
        match offset {
            DukeOffset::Top => or_none(c.y > 0, || Coordinates { x: c.x, y: c.y - 1 }),
            DukeOffset::Bottom => or_none(c.y < self.board.height - 1, || Coordinates { x: c.x, y: c.y + 1 }),
            DukeOffset::Left => or_none(c.x > 0, || Coordinates { x: c.x - 1, y: c.y }),
            DukeOffset::Right => or_none(c.x < self.board.width - 1, || Coordinates { x: c.x + 1, y: c.y }),
        }
    }

    pub fn setup(
        player_1_setup: (DukeInitialLocation, FootmenSetup),
        player_2_setup: (DukeInitialLocation, FootmenSetup),
    ) -> GameBoard {
        let mut result = GameBoard { board: Board::square(GameBoard::BOARD_SIZE) };

        { // First player
            let duke_1_x = match player_1_setup.0 {
                DukeInitialLocation::Left => 3,
                DukeInitialLocation::Right => 2,
            };
            result.place(
                Coordinates { y: 0, x: duke_1_x },
                units::place_tile(Owner::TopPlayer, units::duke),
            );
            let (f1_1, f1_2) = match player_1_setup.1 {
                FootmenSetup::Sides =>
                    (Coordinates { x: duke_1_x + 1, y: 0 }, Coordinates { x: duke_1_x - 1, y: 0 }),
                FootmenSetup::Left =>
                    (Coordinates { x: duke_1_x + 1, y: 0 }, Coordinates { x: duke_1_x, y: 1 }),
                FootmenSetup::Right =>
                    (Coordinates { x: duke_1_x - 1, y: 0 }, Coordinates { x: duke_1_x, y: 1 }),
            };
            result.place(f1_1, units::place_tile(Owner::TopPlayer, units::footman));
            result.place(f1_2, units::place_tile(Owner::TopPlayer, units::footman));
        }
        { // Second player
            let last_row = result.height() - 1;
            let duke_2_x = match player_2_setup.0 {
                DukeInitialLocation::Left => 2,
                DukeInitialLocation::Right => 3,
            };
            result.place(
                Coordinates { y: last_row, x: duke_2_x }, units::place_tile(Owner::BottomPlayer, units::duke));
            let (f2_1, f2_2) = match player_2_setup.1 {
                FootmenSetup::Sides =>
                    (Coordinates { x: duke_2_x + 1, y: last_row }, Coordinates { x: duke_2_x - 1, y: last_row }),
                FootmenSetup::Left =>
                    (Coordinates { x: duke_2_x - 1, y: last_row }, Coordinates { x: duke_2_x, y: last_row - 1 }),
                FootmenSetup::Right =>
                    (Coordinates { x: duke_2_x + 1, y: last_row }, Coordinates { x: duke_2_x, y: last_row - 1 }),
            };
            result.place(f2_1, units::place_tile(Owner::BottomPlayer, units::footman));
            result.place(f2_2, units::place_tile(Owner::BottomPlayer, units::footman));
        }

        result
    }
    pub fn height(&self) -> u16 {
        self.board.height
    }
    pub fn width(&self) -> u16 {
        self.board.width
    }
    pub fn get_board(&self) -> &Board<PlacedTile> {
        &self.board
    }
    pub fn empty() -> GameBoard {
        GameBoard { board: Board::square(GameBoard::BOARD_SIZE) }
    }
    pub fn place(&mut self, c: Coordinates, t: PlacedTile) -> () {
        assert!(self.board.is_empty(c), "Cannot insert tile into occupied space {:?}", c);
        self.board.put(c, t);
    }

    pub fn rows(&self) -> &Vec<Vec<Option<PlacedTile>>> {
        self.board.rows()
    }

    pub fn get(&self, c: Coordinates) -> Option<&PlacedTile> {
        self.board.get(c)
    }

    fn to_absolute_coordinate(
        &self, src: Coordinates, offset: Offsets, center: VerticalOffset,
    ) -> Option<Coordinates> {
        fn vertical_offset(y: VerticalOffset) -> i32 {
            match y {
                VerticalOffset::FarTop => -2,
                VerticalOffset::Top => -1,
                VerticalOffset::Center => 0,
                VerticalOffset::Bottom => 1,
                VerticalOffset::FarBottom => 2,
            }
        }
        let x: i32 = src.x as i32 + match offset.x {
            HorizontalOffset::FarLeft => -2,
            HorizontalOffset::Left => -1,
            HorizontalOffset::Center => 0,
            HorizontalOffset::Right => 1,
            HorizontalOffset::FarRight => 2,
        };
        let y: i32 = src.y as i32 + vertical_offset(offset.y) + vertical_offset(center);
        u16::try_from(x)
            .and_then(|x| u16::try_from(y).map(|y| Coordinates { x, y }))
            .ok()
            .filter(|c| self.board.is_in_bounds(*c))
    }

    fn target_coordinates(
        &self, src: Coordinates, offset: Offsets, action: TileAction, center: VerticalOffset,
    ) -> Vec<Coordinates> {
        match action {
            TileAction::Move => self.to_absolute_coordinate(src, offset, center).into_iter().collect(),
            TileAction::Jump => self.to_absolute_coordinate(src, offset, center).into_iter().collect(),
            TileAction::Strike => self.to_absolute_coordinate(src, offset, center).into_iter().collect(),
            TileAction::Slide => {
                let horizontal = |r: Range<u16>| r.map(|x| Coordinates { x, y: src.y }).collect();
                let vertical = |r: Range<u16>| r.map(|y| Coordinates { x: src.x, y }).collect();
                fn diagonal<I1, I2>(
                    x: I1, y: I2) -> Vec<Coordinates>
                    where I1: Iterator<Item=u16>, I2: Iterator<Item=u16> {
                    x.zip(y).map(|(x, y)| Coordinates { x, y }).collect()
                }
                let res: Vec<Coordinates> =
                    if offset == HorizontalOffset::Right.center() {
                        horizontal(0..src.x)
                    } else if offset == HorizontalOffset::Left.center() {
                        horizontal(src.x + 1..self.width())
                    } else if offset == VerticalOffset::Top.center() {
                        vertical(0..src.y)
                    } else if offset == VerticalOffset::Bottom.center() {
                        vertical(src.y + 1..self.height())
                        // Diagonals
                    } else if offset == Offsets::new(HorizontalOffset::Right, VerticalOffset::Top) {
                        diagonal((0..src.x).rev(), (0..src.y).rev())
                    } else if offset == Offsets::new(HorizontalOffset::Left, VerticalOffset::Top) {
                        diagonal(src.x + 1..self.width(), (0..src.y).rev())
                    } else if offset == Offsets::new(HorizontalOffset::Right, VerticalOffset::Bottom) {
                        diagonal((0..src.x).rev(), src.y + 1..self.height())
                    } else if offset == Offsets::new(HorizontalOffset::Left, VerticalOffset::Bottom) {
                        diagonal(src.x + 1..self.width(), src.y + 1..self.height())
                    } else {
                        panic!("Invalid slide offset {:?}", offset)
                    };
                res.iter().for_each(|e| assert!(self.board.is_in_bounds(*e)));
                res.iter().for_each(|e| assert!(e.is_linear_to(src)));
                res
            }
            TileAction::JumpSlide => unimplemented!(),
            TileAction::Unit => panic!("ASSERTION ERROR"),
            TileAction::Command => panic!("ASSERTION ERROR"),
        }
    }

    fn unobstructed(&self, src: Coordinates, dst: Coordinates) -> bool {
        src.linear_path_to(dst).into_iter().all(|c| self.board.is_empty(c))
    }

    pub fn can_place_new_tile_near_duke(&self, o: Owner) -> bool {
        !self.empty_spaces_near_current_duke(o).is_empty()
    }
    pub fn empty_spaces_near_current_duke(&self, o: Owner) -> Vec<Coordinates> {
        let duke_location = self.duke_coordinates(o);
        DukeOffset::iter()
            .filter_map(|offset| self.absolute_duke_offset(offset, duke_location))
            .filter(|c| self.board.is_empty(*c))
            .collect()
    }

    fn different_team_or_empty(&self, src: Coordinates, dst: Coordinates) -> bool {
        let src_tile = self.board.get(src).expect("No unit found in src to apply an action with");
        self.board.get(dst).for_all(|c| src_tile.different_team(c))
    }

    fn can_apply_action(&self, src: Coordinates, dst: Coordinates, action: &TileAction) -> bool {
        if !self.different_team_or_empty(src, dst) {
            return false;
        }
        match action {
            TileAction::Unit => panic!("Cannot apply action Unit"),
            TileAction::Move => self.unobstructed(src, dst),
            TileAction::Jump => true,
            TileAction::Slide => self.unobstructed(src, dst),
            TileAction::Command => panic!("Commands shouldn't have been used here"),
            TileAction::JumpSlide => todo!(),
            TileAction::Strike => true,
        }
    }
    fn can_apply(
        &self,
        src: Coordinates,
        dst: Coordinates,
    ) -> AppliedPubAction {
        self.get(src)
            .expect("src position is empty")
            .get_action_from_coordinates(src, dst)
            .map_or(
                AppliedPubAction::Invalid,
                |a| {
                    match (a, self.can_apply_action(src, dst, a)) {
                        (_, false) => AppliedPubAction::Invalid,
                        (TileAction::Strike, true) => AppliedPubAction::Strike,
                        (_, true) => AppliedPubAction::Movement,
                    }
                },
            )
    }

    fn can_command(
        &self,
        commander_src: Coordinates,
        unit_src: Coordinates,
        unit_dst: Coordinates,
    ) -> bool {
        let commander_tile = self.get(commander_src).expect("No unit found in commander_src");
        let unit_tile = self.get(unit_src).expect("No commanded unit found in unit_src");
        assert!(commander_tile.same_team(unit_tile), "Cannot command a unit from a different team");
        self.different_team_or_empty(unit_src, unit_dst)
    }

    pub fn duke_coordinates(&self, o: Owner) -> Coordinates {
        self.board
            .find(|a| a.owner == o && units::is_duke(a.tile.borrow()))
            .expect(format!("Could not find the duke for {:?}", o).as_str())
    }

    fn flip(&mut self, c: Coordinates) -> () {
        self.board.get_mut(c).unwrap().flip()
    }

    pub fn can_move(&self, src: Coordinates, dst: Coordinates) -> bool {
        self.can_apply(src, dst) != AppliedPubAction::Invalid
    }

    // TODO: Should this really accept an owner?
    // Returns true if succeeded. This allows GameMove to be move instead of being borrowed, which
    // the types all around nicer to use
    pub fn make_a_move(&mut self, gm: GameMove, o: Owner) -> () {
        match gm {
            GameMove::PlaceNewTile(tile, duke_offset) => {
                let c = self.absolute_duke_offset(duke_offset, self.duke_coordinates(o))
                    .expect("Request duke location is out of bounds");
                assert!(self.board.is_empty(c), "Cannot place new tile in non-empty spot");
                self.place(c, PlacedTile::new(o.clone(), tile.clone()));
            }
            GameMove::ApplyNonCommandTileAction { src, dst } => {
                let tile = self.board.get(src).expect("Cannot move from an empty tile");
                assert_eq!(
                    tile.owner,
                    o,
                    "Cannot move unowned tile in {:?}",
                    src
                );
                match self.can_apply(src, dst) {
                    AppliedPubAction::Movement => {
                        self.flip(src);
                        self.board.mv(src, dst);
                    }
                    AppliedPubAction::Strike => {
                        self.flip(src);
                        self.board.remove(dst);
                    }
                    AppliedPubAction::Invalid =>
                        panic!("Cannot move unit in {:?} to {:?}", &src, &dst)
                }
            }

            GameMove::CommandAnotherTile { commander_src, unit_src, unit_dst } => {
                let commander = self.board.get(commander_src).expect("Cannot command from an empty tile");
                assert_eq!(
                    commander.owner,
                    o,
                    "Cannot command using unowned command in {:?}",
                    commander_src
                );
                assert!(
                    self.can_command(commander_src, unit_src, unit_dst),
                    "Can't apply command (commander: {:?}, unit_src: {:?}, unit_dst: {:?}",
                    commander_src, unit_src, unit_dst,
                );
                self.flip(commander_src);
                self.board.mv(unit_src, unit_dst);
            }
        }
    }

    pub fn get_tiles_for(&self, o: Owner) -> Vec<(Coordinates, &PlacedTile)> {
        self.board
            .active_coordinates()
            .into_iter()
            .filter(|e| e.1.owner.same_team(o))
            .collect()
    }

    // Except commands.
    pub fn get_legal_moves(&self, src: Coordinates) -> Vec<(Coordinates, TileAction)> {
        let tile = self.get(src).unwrap().get_current_side();
        let center_offset = tile.center_offset();
        tile.actions()
            .into_iter()
            .filter(|e| e.1 != TileAction::Command && e.1 != TileAction::Unit && e.1 != TileAction::Jump)
            .flat_map(|o| self
                .target_coordinates(src, o.0, o.1, center_offset)
                .iter()
                .map(|c| (*c, o.1))
                .collect::<Vec<(Coordinates, TileAction)>>()
            )
            .filter(|o| self.can_apply_action(src, o.0, &o.1))
            .collect()
    }
}

#[cfg(test)]
mod test {
    use crate::assert_eq_set;

    use super::*;

    #[test]
    fn get_legal_moves_moves_only() {
        let mut gs = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        gs.place(c, units::place_tile(Owner::TopPlayer, units::footman));
        assert_eq_set!(
            vec![
                (Coordinates { x: 3, y: 4 }, TileAction::Move),
                (Coordinates { x: 2, y: 5 }, TileAction::Move),
                (Coordinates { x: 1, y: 4 }, TileAction::Move),
                (Coordinates { x: 2, y: 3 }, TileAction::Move),
            ],
            gs.get_legal_moves(c),
        );
    }

    #[test]
    fn get_legal_moves_horizontal_slides() {
        let mut gs = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        gs.place(c, units::place_tile(Owner::TopPlayer, units::duke));
        assert_eq_set!(
            vec![
                (Coordinates { x: 0, y: 4 }, TileAction::Slide),
                (Coordinates { x: 1, y: 4 }, TileAction::Slide),
                (Coordinates { x: 3, y: 4 }, TileAction::Slide),
                (Coordinates { x: 4, y: 4 }, TileAction::Slide),
                (Coordinates { x: 5, y: 4 }, TileAction::Slide),
            ],
            gs.get_legal_moves(c),
        );
    }

    #[test]
    fn get_legal_moves_vertical_slides() {
        let mut gs = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        gs.place(c, units::place_tile(Owner::TopPlayer, units::duke));
        gs.flip(c);
        assert_eq_set!(
            vec![
                (Coordinates { x: 2, y: 0 }, TileAction::Slide),
                (Coordinates { x: 2, y: 1 }, TileAction::Slide),
                (Coordinates { x: 2, y: 2 }, TileAction::Slide),
                (Coordinates { x: 2, y: 3 }, TileAction::Slide),
                (Coordinates { x: 2, y: 5 }, TileAction::Slide),
            ],
            gs.get_legal_moves(c),
        );
    }

    #[test]
    fn get_legal_moves_diagonal_slides() {
        let mut gs = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        gs.place(c, units::place_tile(Owner::TopPlayer, units::priest));
        assert_eq_set!(
            vec![
                (Coordinates { x: 1, y: 3 }, TileAction::Slide),
                (Coordinates { x: 0, y: 2 }, TileAction::Slide),

                (Coordinates { x: 3, y: 5 }, TileAction::Slide),

                (Coordinates { x: 1, y: 5 }, TileAction::Slide),

                (Coordinates { x: 3, y: 3 }, TileAction::Slide),
                (Coordinates { x: 4, y: 2 }, TileAction::Slide),
                (Coordinates { x: 5, y: 1 }, TileAction::Slide),
            ],
            gs.get_legal_moves(c),
        );
    }


    #[test]
    fn make_a_move() {
        let mut gs = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        gs.place(c, units::place_tile(Owner::TopPlayer, units::footman));
        let c2 = Coordinates { x: 1, y: 4 };
        gs.make_a_move(
            GameMove::ApplyNonCommandTileAction { src: c, dst: c2 },
            Owner::TopPlayer,
        );
        assert!(gs.get(c2).is_some());
        assert!(gs.get(c).is_none());
    }
}
