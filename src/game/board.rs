use std::borrow::Borrow;
use std::convert::TryFrom;
use std::ops::Range;

use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use crate::common::board::Board;
use crate::common::coordinates::Coordinates;
use crate::common::geometry::Rectangular;
use crate::common::utils::Folding;
use crate::game::dumb_printer::{double_char_print_board, single_char_print_board};
use crate::game::offset::{Centerable, HorizontalOffset, Offsets, VerticalOffset};
use crate::game::tile::{Owner, Ownership, PlacedTile, TileRef};
use crate::game::tile_side::TileAction;
use crate::game::units;
use crate::time_it_macro;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, EnumIter)]
pub enum DukeOffset { Top, Bottom, Left, Right }

#[derive(Debug, Clone)]
pub(super) enum BoardMove {
    PlaceNewTile(TileRef, DukeOffset, Owner),
    ApplyNonCommandTileAction { src: Coordinates, dst: Coordinates },
    // CommandAnotherTile { commander_src: Coordinates, unit_src: Coordinates, unit_dst: Coordinates },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PossibleMove {
    PlaceNewTile(DukeOffset, Owner),
    ApplyNonCommandTileAction { src: Coordinates, dst: Coordinates, capturing: Option<PlacedTile> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GameBoard {
    board: Board<PlacedTile>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WithNewTiles(pub bool);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CheckForGuard(pub bool);

impl CheckForGuard {
    pub fn if_check(&self, f: impl FnOnce() -> bool) -> bool {
        !self.0 || f()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppliedPubAction { Movement, Strike, Invalid }

impl GameBoard {
    pub const BOARD_SIZE: u16 = 6;

    pub(super) fn new(board: Board<PlacedTile>) -> Self { GameBoard { board } }
    fn absolute_duke_offset(&self, offset: DukeOffset, c: Coordinates) -> Option<Coordinates> {
        fn or_none<P>(b: bool, c: P) -> Option<Coordinates> where P: Fn() -> Coordinates {
            if b { Some(c()) } else { None }
        }
        match offset {
            DukeOffset::Top => or_none(c.y > 0, || Coordinates { x: c.x, y: c.y - 1 }),
            DukeOffset::Bottom => or_none(c.y < self.height() - 1, || Coordinates { x: c.x, y: c.y + 1 }),
            DukeOffset::Left => or_none(c.x > 0, || Coordinates { x: c.x - 1, y: c.y }),
            DukeOffset::Right => or_none(c.x < self.width() - 1, || Coordinates { x: c.x + 1, y: c.y }),
        }
    }

    pub fn to_absolute_duke_offset(&self, offset: DukeOffset, o: Owner) -> Option<Coordinates> {
        self.absolute_duke_offset(offset, self.duke_coordinates(o))
    }

    pub fn get_board(&self) -> &Board<PlacedTile> {
        &self.board
    }

    #[cfg(test)]
    pub fn empty() -> GameBoard {
        GameBoard { board: Board::square(GameBoard::BOARD_SIZE) }
    }
    pub fn place(&mut self, c: Coordinates, t: PlacedTile) -> () {
        assert!(self.board.is_empty(c), "Cannot insert tile into occupied space {:?}", c);
        self.board.put(c, t);
    }
    fn remove(&mut self, c: Coordinates) -> PlacedTile {
        self.board.remove(c).expect(format!("Cannot remove tile from empty space {:?}", c).as_str())
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

    // TODO this should return an Iterator
    fn target_coordinates(
        &self, src: Coordinates, offset: Offsets, action: TileAction, center: VerticalOffset,
    ) -> Vec<Coordinates> {
        macro_rules! to_list {
            ($o: expr) => {match $o {
                None => Vec::new(),
                Some(x) => vec![x],
            }
            }
        }
        match action {
            TileAction::Move => to_list!(self.to_absolute_coordinate(src, offset, center)),
            TileAction::Jump => to_list!(self.to_absolute_coordinate(src, offset, center)),
            TileAction::Strike => to_list!(self.to_absolute_coordinate(src, offset, center)),
            TileAction::Slide => {
                let horizontal = |r: Range<u16>| r.map(|x| Coordinates { x, y: src.y }).collect();
                let vertical = |r: Range<u16>| r.map(|y| Coordinates { x: src.x, y }).collect();
                fn diagonal<I1, I2>(x: I1, y: I2) -> Vec<Coordinates>
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
                if cfg!(debug_assertions) {
                    res.iter().for_each(|e| assert!(self.board.is_in_bounds(*e)));
                    res.iter().for_each(|e| assert!(e.is_straight_line_to(src)));
                }
                res
            }
            TileAction::JumpSlide => unimplemented!(),
            TileAction::Unit => panic!("ASSERTION ERROR"),
            TileAction::Command => panic!("ASSERTION ERROR"),
        }
    }

    fn unobstructed(&self, src: Coordinates, dst: Coordinates) -> bool {
        if src.is_near(dst) {
            self.board.is_empty(dst)
        } else {
            !src.on_the_linear_path_to(dst, |x, y| self.board.is_occupied(Coordinates { x, y }))
        }
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

    #[inline(always)]
    fn different_team_or_empty(&self, src: Coordinates, dst: Coordinates) -> bool {
        let src_tile = self.board.get(src).expect("No unit found in src to apply an action with");
        self.board.get(dst).for_all(|c| src_tile.different_team(c))
    }

    fn can_apply_action(&self, src: Coordinates, dst: Coordinates, action: TileAction) -> bool {
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
            TileAction::Strike => self.get(dst).exists(|o| o.different_team(&self.get(src).unwrap())),
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

    // fn can_command(
//     &self,
//     commander_src: Coordinates,
//     unit_src: Coordinates,
//     unit_dst: Coordinates,
// ) -> bool {
//     let commander_tile = self.get(commander_src).expect("No unit found in commander_src");
//     let unit_tile = self.get(unit_src).expect("No commanded unit found in unit_src");
//     assert!(commander_tile.same_team(unit_tile), "Cannot command a unit from a different team");
//     self.different_team_or_empty(unit_src, unit_dst)
// }
//
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

    pub fn is_valid_placement(&self, owner: Owner, offset: DukeOffset) -> bool {
        self.is_valid_placement_aux(owner, offset, CheckForGuard(true))
    }

    fn is_valid_placement_aux(&self, owner: Owner, offset: DukeOffset, cfg: CheckForGuard) -> bool {
        match self.absolute_duke_offset(offset, self.duke_coordinates(owner)) {
            None => false,
            Some(c) =>
                if self.board.is_occupied(c) {
                    false
                } else {
                    cfg.if_check(|| {
                        // Cannot use does_not_put_in_guard as that will cause an infinite recursion.
                        // TODO cache this footman, stop cloning for guard checks.
                        let mut clone = self.clone();
                        clone.place(c, PlacedTile::new(owner, units::footman()));
                        !clone.is_guard(owner)
                    })
                }
        }
    }

    pub(super) fn make_a_move(&mut self, gm: BoardMove) -> () {
        match gm {
            BoardMove::PlaceNewTile(tile, duke_offset, owner) => {
                let c = self.absolute_duke_offset(duke_offset, self.duke_coordinates(owner))
                    .expect("Request duke location is out of bounds");
                assert!(self.is_valid_placement(owner, duke_offset));
                self.place(c, PlacedTile::new_from_ref(owner, tile));
            }
            BoardMove::ApplyNonCommandTileAction { src, dst } => {
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

            // BoardMove::CommandAnotherTile { commander_src, unit_src, unit_dst } => {
            //     let commander = self.board.get(commander_src).expect("Cannot command from an empty tile");
            //     assert_eq!(
            //         commander.owner,
            //         o,
            //         "Cannot command using unowned command in {:?}",
            //         commander_src
            //     );
            //     assert!(
            //         self.can_command(commander_src, unit_src, unit_dst),
            //         "Can't apply command (commander: {:?}, unit_src: {:?}, unit_dst: {:?}",
            //         commander_src, unit_src, unit_dst,
            //     );
            //     self.flip(commander_src);
            //     self.board.mv(unit_src, unit_dst);
            // }
        }
    }

    // TODO should also return an iterator
    pub fn get_tiles_for(&self, o: Owner) -> Vec<(Coordinates, &PlacedTile)> {
        self.board
            .active_coordinates()
            .into_iter()
            .filter(|e| e.1.owner.same_team(&o))
            .collect()
    }

    // Except commands.
    pub fn get_legal_moves(&self, src: Coordinates) -> Vec<(Coordinates, TileAction)> {
        self.get_legal_moves_aux(src, CheckForGuard(true)).collect()
    }

    pub fn get_legal_moves_ignoring_guard(&self, src: Coordinates) -> Vec<(Coordinates, TileAction)> {
        self.get_legal_moves_aux(src, CheckForGuard(false)).collect()
    }

    fn get_legal_moves_aux(
        &self, src: Coordinates, cfg: CheckForGuard) -> Box<dyn Iterator<Item=(Coordinates, TileAction)> + '_> {
        let tile = self.get(src).unwrap();
        let owner = tile.owner;
        let tile_side = tile.get_current_side();
        let center_offset = tile_side.center_offset();
        Box::new(
            tile_side.actions()
                .iter()
                .filter(|e| e.1 != TileAction::Command && e.1 != TileAction::Unit)
                .flat_map(move |o| self
                    .target_coordinates(src, o.0, o.1, center_offset)
                    .into_iter()
                    .map(move |c| (c, o.1))
                )
                .filter(move |o| self.can_apply_action(src, o.0, o.1))
                .filter(move |o| cfg.if_check(||
                    self.does_not_put_in_guard(
                        BoardMove::ApplyNonCommandTileAction { src, dst: o.0 },
                        owner,
                    )
                ))
                .into_iter()
        )
    }

    pub fn is_guard(&self, owner: Owner) -> bool {
        time_it_macro!("is_guard", {
            let c = self.duke_coordinates(owner);
            self.get_board()
                .active_coordinates()
                .filter(|e| e.1.owner.different_team(&owner))
                .any(|other_tile|
                    self
                        .get_legal_moves_aux(other_tile.0, CheckForGuard(false))
                        .any(|other_move| other_move.0 == c)
                )
        })
    }

    pub(super) fn does_not_put_in_guard(&self, mv: BoardMove, owner: Owner) -> bool {
        let mut clone = self.clone();
        clone.make_a_move(mv);
        !clone.is_guard(owner)
    }

    // Returns the tile that was removed, if such a tile exists, e.g., when placing a new tile,
    // undoing the action would remove the new tile from the board.
    pub fn undo(&mut self, mv: PossibleMove) -> Option<PlacedTile> {
        match mv {
            PossibleMove::PlaceNewTile(offset, owner) => {
                let absolute_coordinate = self
                    .to_absolute_duke_offset(offset, owner)
                    .expect(format!(
                        "Invalid tile placement {:?} relative to duke {:?}",
                        offset,
                        self.duke_coordinates(owner),
                    ).as_str());
                Some(self.remove(absolute_coordinate))
            }
            PossibleMove::ApplyNonCommandTileAction { src, dst, capturing } => {
                if !self.board.is_occupied(dst) { // Strike
                    let captured = capturing.expect("No captured but attacker didn't move");
                    self.flip(src);
                    self.place(dst, captured);
                } else {
                    let mut mover = self.remove(dst);
                    mover.flip();
                    self.place(src, mover);
                    if let Some(captured) = capturing {
                        self.place(dst, captured.clone());
                    }
                }
                None
            }
        }
    }

    pub fn all_valid_moves(&self, owner: Owner, new_tiles: WithNewTiles) -> Box<dyn Iterator<Item=PossibleMove> + '_> {
        self.all_valid_moves_aux(owner, new_tiles, CheckForGuard(true))
    }

    pub fn all_valid_moves_ignoring_guard(&self, owner: Owner, new_tiles: WithNewTiles) -> Box<dyn Iterator<Item=PossibleMove> + '_> {
        self.all_valid_moves_aux(owner, new_tiles, CheckForGuard(false))
    }

    fn all_valid_moves_aux(
        &self,
        owner: Owner,
        new_tiles: WithNewTiles,
        cfg: CheckForGuard,
    ) -> Box<dyn Iterator<Item=PossibleMove> + '_> {
        let result = self
            .get_tiles_for(owner)
            .into_iter()
            .map(|e| e.0)
            .flat_map(move |src| self
                .get_legal_moves_aux(src, cfg)
                .map(move |e| e.0)
                .map(move |dst| PossibleMove::ApplyNonCommandTileAction {
                    src,
                    dst,
                    capturing: self.board.get(dst).cloned(),
                })
                .collect::<Vec<_>>()
            );

        if let WithNewTiles(true) = new_tiles {
            Box::new(result.chain(
                DukeOffset::iter().filter_map(move |offset|
                    if self.is_valid_placement_aux(owner, offset, cfg) {
                        Some(PossibleMove::PlaceNewTile(offset, owner))
                    } else {
                        None
                    })
            )
            )
        } else {
            Box::new(result)
        }
    }

    #[allow(dead_code)]
    pub fn as_single_string(&self) -> String { single_char_print_board(&self.board) }
    #[allow(dead_code)]
    pub fn as_double_string(&self) -> String { double_char_print_board(&self.board) }
    #[allow(dead_code)]
    pub fn debug_double(&self) { println!("{}", self.as_double_string()); }
}

impl Rectangular for GameBoard {
    fn width(&self) -> u16 {
        self.board.width()
    }

    fn height(&self) -> u16 {
        self.board.height()
    }
}

#[cfg(test)]
mod test {
    use crate::{assert_empty, assert_eq_set, assert_not};

    use super::*;

    // get_legal_moves
    #[test]
    fn get_legal_moves_moves_only() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));

        let c = Coordinates { x: 2, y: 4 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::footman));
        assert_eq_set!(
            vec![
                (Coordinates { x: 3, y: 4 }, TileAction::Move),
                (Coordinates { x: 2, y: 5 }, TileAction::Move),
                (Coordinates { x: 1, y: 4 }, TileAction::Move),
                (Coordinates { x: 2, y: 3 }, TileAction::Move),
            ],
            board.get_legal_moves(c),
        );
    }

    #[test]
    fn get_legal_moves_and_jumps() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 1, y: 4 };
        board.place(Coordinates { x: 5, y: 5 }, units::place_tile(Owner::TopPlayer, units::duke));
        board.place(c, units::place_tile(Owner::TopPlayer, units::champion));
        assert_eq_set!(
            vec![
                (Coordinates { x: 1, y: 2 }, TileAction::Jump),
                (Coordinates { x: 1, y: 3 }, TileAction::Move),
                (Coordinates { x: 1, y: 5 }, TileAction::Move),
                (Coordinates { x: 0, y: 4 }, TileAction::Move),
                (Coordinates { x: 2, y: 4 }, TileAction::Move),
                (Coordinates { x: 3, y: 4 }, TileAction::Jump),
            ],
            board.get_legal_moves(c),
        );
    }

    #[test]
    fn get_legal_moves_horizontal_slides() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::duke));
        assert_eq_set!(
            vec![
                (Coordinates { x: 0, y: 4 }, TileAction::Slide),
                (Coordinates { x: 1, y: 4 }, TileAction::Slide),
                (Coordinates { x: 3, y: 4 }, TileAction::Slide),
                (Coordinates { x: 4, y: 4 }, TileAction::Slide),
                (Coordinates { x: 5, y: 4 }, TileAction::Slide),
            ],
            board.get_legal_moves(c),
        );
    }

    #[test]
    fn get_legal_moves_vertical_slides() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        board.place(c, units::place_tile_flipped(Owner::TopPlayer, units::duke));
        assert_eq_set!(
            vec![
                (Coordinates { x: 2, y: 0 }, TileAction::Slide),
                (Coordinates { x: 2, y: 1 }, TileAction::Slide),
                (Coordinates { x: 2, y: 2 }, TileAction::Slide),
                (Coordinates { x: 2, y: 3 }, TileAction::Slide),
                (Coordinates { x: 2, y: 5 }, TileAction::Slide),
            ],
            board.get_legal_moves(c),
        );
    }

    #[test]
    fn can_move_returns_true_for_diagonal_sliding() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::priest));
        assert!(board.can_move(c, Coordinates { x: 4, y: 2 }));
    }

    #[test]
    fn can_move_returns_true_for_capture() {
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 2, y: 4 };
        let dst = Coordinates { x: 2, y: 5 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::footman));
        board.place(dst, units::place_tile(Owner::BottomPlayer, units::footman));
        assert!(board.can_move(src, dst));
    }

    #[test]
    fn can_move_returns_false_for_occupied_with_same() {
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 2, y: 4 };
        let dst = Coordinates { x: 2, y: 5 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::footman));
        board.place(dst, units::place_tile(Owner::TopPlayer, units::footman));
        assert_not!(board.can_move(src, dst));
    }

    #[test]
    fn can_move_returns_false_for_movement_out_of_scope() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 2, y: 2 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::footman));
        assert_not!(board.can_move(c, Coordinates { x: 5, y: 5 }));
    }

    #[test]
    fn can_move_returns_false_for_empty_strike() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 0, y: 0 };
        let mut tile = units::place_tile(Owner::TopPlayer, units::pikeman);
        tile.flip();
        board.place(c, tile);
        assert_not!(board.can_move(c, Coordinates { x: 1, y: 2 }));
    }

    #[test]
    fn get_legal_moves_diagonal_slides() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));

        let c = Coordinates { x: 2, y: 4 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::priest));
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
            board.get_legal_moves(c),
        );
    }

    #[test]
    fn is_guard_takes_strikes_into_account() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));
        let coordinates = Coordinates { x: 1, y: 2 };
        board.place(coordinates, units::place_tile_flipped(Owner::BottomPlayer, units::pikeman));
        assert!(board.is_guard(Owner::TopPlayer));
    }

    // make_a_move
    #[test]
    fn make_a_move() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::footman));
        let c2 = Coordinates { x: 1, y: 4 };
        board.make_a_move(
            BoardMove::ApplyNonCommandTileAction { src: c, dst: c2 },
        );
        assert!(board.get(c2).is_some());
        assert!(board.get(c).is_none());
    }

    #[test]
    fn is_valid_placement_returns_false_if_in_guard() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 0, y: 0 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::duke));
        let c2 = Coordinates { x: 0, y: 2 };
        board.place(c2, units::place_tile_flipped(Owner::BottomPlayer, units::footman));
        assert_not!(board.is_valid_placement(Owner::TopPlayer, DukeOffset::Right));
    }

    #[test]
    fn get_legal_moves_can_block_guard() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let mut footman = PlacedTile::new(Owner::TopPlayer, units::footman());
        footman.flip();
        let footman_coordinates = Coordinates { x: 4, y: 5 };
        board.place(footman_coordinates, footman);
        let mut op_duke = PlacedTile::new(Owner::BottomPlayer, units::duke());
        op_duke.flip();
        board.place(Coordinates { x: 5, y: 0 }, op_duke);

        // TopPlayer can still play a footman move
        assert_eq!(
            board.get_legal_moves(footman_coordinates),
            vec![(Coordinates { x: 5, y: 4 }, TileAction::Move)],
        )
    }

    #[test]
    fn get_legal_moves_does_not_allow_placing_duke_in_strike() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, units::place_tile(Owner::TopPlayer, units::duke));
        let pikeman_coordinates = Coordinates { x: 2, y: 2 };
        board.place(pikeman_coordinates, units::place_tile_flipped(Owner::BottomPlayer, units::pikeman));
        board.place(Coordinates { x: 1, y: 0 }, units::place_tile(Owner::BottomPlayer, units::footman));
        assert_empty!(board.get_legal_moves(duke_coordinates));
    }
}
