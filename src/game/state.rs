use std::hash::{Hash, Hasher};
use std::sync::Arc;
use rand::seq::SliceRandom;
use rand::Rng;
use strum::IntoEnumIterator;

use crate::assert_not;
use crate::common::board::Board;
use crate::common::coordinates::Coordinates;
use crate::common::geometry::Rectangular;
use crate::common::percentage::Percentage;
use crate::game::{board_setup, units};
use crate::game::bag::{DiscardBag, TileBag};
use crate::game::board::{BoardMove, DukeOffset, GameBoard, PossibleMove, WithNewTiles};
use crate::game::board_setup::{DukeInitialLocation, FootmenSetup};
use crate::game::dumb_printer::{double_char_print_state, single_char_print_state};
use crate::game::tile::{CurrentSide, Owner, PlacedTile, TileRef};
use crate::game::tile_side::TileAction;

// Technically not part of the base game rules, but it makes it easier for the AI
pub const MAX_MOVES_WITHOUT_CAPTURE_OR_PLACEMENT: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameState {
    board: GameBoard,
    pulled_tile: Option<TileRef>,
    current_player_turn: Owner,
    top_player_bag: TileBag,
    top_player_discard: DiscardBag,
    bottom_player_bag: TileBag,
    bottom_player_discard: DiscardBag,
    moves_without_capture_or_placement_stack: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum GameMove {
    // TODO document how you can place without pulling...
    PlaceNewTile(DukeOffset),
    PullAndPlay(DukeOffset),
    ApplyNonCommandTileAction { src: Coordinates, dst: Coordinates },
}

impl Into<GameMove> for &PossibleMove {
    fn into(self) -> GameMove {
        match self {
            PossibleMove::PlaceNewTile(o, _) => GameMove::PullAndPlay(*o),
            PossibleMove::ApplyNonCommandTileAction { src, dst, .. } =>
                GameMove::ApplyNonCommandTileAction {
                    src: *src,
                    dst: *dst,
                }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CanPullNewTileResult {
    EmptyBag,
    NoSpaceNearDuke,
    DukeAlwaysInGuard,
    OK,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum GameResult {
    Tie,
    Ongoing,
    Won(Owner),
}

impl GameState {
    pub fn board(&self) -> &Board<PlacedTile> { self.board.get_board() }

    pub fn pulled_tile(&self) -> &Option<TileRef> { &self.pulled_tile }
    pub fn current_player_turn(&self) -> Owner { self.current_player_turn }
    pub fn top_player_bag(&self) -> &TileBag { &self.top_player_bag }
    pub fn player_1_discard(&self) -> &DiscardBag { &self.top_player_discard }
    pub fn bottom_player_bag(&self) -> &TileBag { &self.bottom_player_bag }
    pub fn player_2_discard(&self) -> &DiscardBag { &self.bottom_player_discard }

    #[cfg(test)]
    pub(super) fn from_board(board: GameBoard, current_player: Owner) -> GameState {
        GameState::from_board_with_bag(board, current_player, TileBag::empty())
    }
    #[cfg(test)]
    pub(super) fn from_board_with_bag(
        board: GameBoard, current_player_turn: Owner, bag: TileBag) -> GameState {
        GameState {
            board,
            current_player_turn,
            pulled_tile: None,
            top_player_bag: bag.clone(),
            top_player_discard: DiscardBag::empty(),
            bottom_player_bag: bag,
            bottom_player_discard: DiscardBag::empty(),
            moves_without_capture_or_placement_stack: vec![0],
        }
    }
    pub fn new(
        base_bag: &TileBag,
        player_1_setup: (DukeInitialLocation, FootmenSetup),
        player_2_setup: (DukeInitialLocation, FootmenSetup),
    ) -> GameState {
        let board = board_setup::setup(player_1_setup, player_2_setup);

        GameState {
            board,
            current_player_turn: Owner::TopPlayer,
            pulled_tile: None,
            top_player_bag: base_bag.clone(),
            top_player_discard: DiscardBag::empty(),
            bottom_player_bag: base_bag.clone(),
            bottom_player_discard: DiscardBag::empty(),
            moves_without_capture_or_placement_stack: vec![0],
        }
    }

    fn can_pull_tile_from_bag(&self) -> CanPullNewTileResult {
        let bag = match self.current_player_turn {
            Owner::TopPlayer => &self.top_player_bag,
            Owner::BottomPlayer => &self.bottom_player_bag,
        };
        if bag.is_empty() {
            CanPullNewTileResult::EmptyBag
        } else if self.board.can_place_new_tile_near_duke(self.current_player_turn) {
            if DukeOffset::iter().any(|offset| self.is_valid_placement(offset)) {
                CanPullNewTileResult::OK
            } else {
                CanPullNewTileResult::DukeAlwaysInGuard
            }
        } else {
            CanPullNewTileResult::NoSpaceNearDuke
        }
    }

    pub fn can_pull_tile_from_bag_bool(&self) -> bool {
        self.can_pull_tile_from_bag() == CanPullNewTileResult::OK
    }

    pub fn pull_tile_from_bag<R: Rng>(&mut self, rng: &mut R) -> () {
        let can_pull = self.can_pull_tile_from_bag();
        assert_eq!(
            can_pull, CanPullNewTileResult::OK, "Cannot pull new tile from bag: {:?}", can_pull);
        let bag = match self.current_player_turn {
            Owner::TopPlayer => &mut self.top_player_bag,
            Owner::BottomPlayer => &mut self.bottom_player_bag,
        };
        let result =
            bag.pull(rng).expect("Assertion Error: bag should not have been empty by this point");
        self.pulled_tile = Some(result);
    }

    fn is_waiting_for_tile_placement(&self) -> bool {
        self.pulled_tile.is_some()
    }

    pub fn can_make_a_move(&self, game_move: &GameMove) -> bool {
        match game_move {
            GameMove::PlaceNewTile(o) =>
                self.is_waiting_for_tile_placement() && self.is_valid_placement(*o),
            GameMove::PullAndPlay(o) =>
                self.can_pull_tile_from_bag_bool() && self.is_valid_placement(*o),
            GameMove::ApplyNonCommandTileAction { src, dst } =>
                self.board.can_move(*src, *dst) &&
                    self.board.does_not_put_in_guard(
                        self.game_move_to_board_move(&game_move), self.current_player_turn),
        }
    }
    pub fn make_a_move<R: Rng>(&mut self, game_move: GameMove, rng: &mut R) -> () {
        match game_move {
            GameMove::PlaceNewTile(_) | GameMove::PullAndPlay(_) =>
                self.moves_without_capture_or_placement_stack.push(0),
            GameMove::ApplyNonCommandTileAction { src, dst } =>
                if self.board.can_move(src, dst) && self.board.get(dst).is_some() {
                    self.moves_without_capture_or_placement_stack.push(0)
                } else {
                    *self.moves_without_capture_or_placement_stack.last_mut().unwrap() += 1
                },
        };
        if let GameMove::PlaceNewTile(_) = game_move {
            assert!(self.is_waiting_for_tile_placement(), "Invalid state for placing a new tile");
        } else {
            assert_not!(self.is_waiting_for_tile_placement(), "Waiting for a new tile placement");
        }
        if let GameMove::PullAndPlay(o) = &game_move {
            self.pull_tile_from_bag(rng);
            self.make_a_move(GameMove::PlaceNewTile(*o), rng);
            return;
        }
        if let GameMove::ApplyNonCommandTileAction { src, dst } = game_move {
            let tile = self.board.get(src).expect("Cannot move from an empty tile");
            assert_eq!(
                tile.owner,
                self.current_player_turn,
                "Cannot move unowned tile in {:?}",
                src
            );
            assert!(self.board.can_move(src, dst), "Can't move from {} to {}", src, dst)
        }
        self.board.make_a_move(self.game_move_to_board_move(&game_move));
        assert_not!(self.board.is_guard(self.current_player_turn));
        self.current_player_turn = self.current_player_turn.next_player();
        if self.is_waiting_for_tile_placement() {
            self.pulled_tile = None
        }
    }

    fn game_move_to_board_move(&self, gm: &GameMove) -> BoardMove {
        match gm {
            GameMove::PlaceNewTile(offset) =>
                BoardMove::PlaceNewTile(
                    self.pulled_tile.as_ref().expect("No pulled tile").clone(),
                    *offset,
                    self.current_player_turn,
                ),
            GameMove::ApplyNonCommandTileAction { src, dst } =>
                BoardMove::ApplyNonCommandTileAction { src: *src, dst: *dst },
            // GameMove::CommandAnotherTile { commander_src, unit_src, unit_dst } =>
            //     BoardMove::CommandAnotherTile { commander_src, unit_src, unit_dst },
            GameMove::PullAndPlay(_) => todo!(),
        }
    }

    fn unit_stub() -> TileRef { Arc::new(units::footman()) }
    fn possible_move_to_board_move(&self, pm: &PossibleMove) -> BoardMove {
        match pm {
            PossibleMove::PlaceNewTile(offset, owner) => BoardMove::PlaceNewTile(
                GameState::unit_stub(),
                *offset,
                *owner,
            ),
            PossibleMove::ApplyNonCommandTileAction { src, dst, .. } =>
                BoardMove::ApplyNonCommandTileAction { src: *src, dst: *dst },
        }
    }


    pub fn get_tiles_for_current_owner(&self) -> Vec<(Coordinates, &PlacedTile)> {
        self.get_tiles_for_owner(self.current_player_turn)
    }

    pub fn get_tiles_for_owner(&self, o: Owner) -> Vec<(Coordinates, &PlacedTile)> {
        self.board.get_tiles_for(o)
    }

    // Except commands
    pub fn get_legal_moves(&self, src: Coordinates) -> Vec<(Coordinates, TileAction)> {
        self.board.get_legal_moves(src)
    }

    // Except commands
    pub fn get_legal_moves_ignoring_guard(&self, src: Coordinates) -> Vec<(Coordinates, TileAction)> {
        self.board.get_legal_moves_ignoring_guard(src)
    }

    pub fn current_duke_coordinate(&self) -> Coordinates {
        self.duke_coordinate(self.current_player_turn)
    }

    pub fn duke_coordinate(&self, o: Owner) -> Coordinates {
        self.board.duke_coordinates(o)
    }

    pub fn empty_spaces_near_current_duke(&self) -> Vec<Coordinates> {
        self.board.empty_spaces_near_current_duke(self.current_player_turn)
    }

    pub fn is_valid_placement(&self, offset: DukeOffset) -> bool {
        let owner = self.current_player_turn;
        self.board.is_valid_placement(owner, offset) &&
            self.board.does_not_put_in_guard(
                BoardMove::PlaceNewTile(TileRef::new(units::footman()), offset, owner),
                owner,
            )
    }

    pub fn is_tie(&self) -> bool {
        self.moves_without_capture_or_placement_stack.last().unwrap() >=
            &MAX_MOVES_WITHOUT_CAPTURE_OR_PLACEMENT
    }
    pub fn is_over(&self) -> bool {
        self.all_valid_game_moves_for_current_player().next().is_none() || self.is_tie()
    }

    pub fn game_result(&self) -> GameResult {
        if self.is_tie() {
            GameResult::Tie
        } else if self.is_over() {
            GameResult::Won(self.current_player_turn.next_player())
        } else {
            GameResult::Ongoing
        }
    }

    // Except commands for now
    pub fn all_valid_game_moves_for_current_player(&self) -> impl Iterator<Item=PossibleMove> + '_ {
        self.all_valid_game_moves_for(self.current_player_turn)
    }

    // Faster than collecting the above, since it avoid some validations.
    pub fn get_random_move_for_current_player<R>(
        &self, rng: &mut R, new_tile_boost: Percentage,
    ) -> Option<PossibleMove> where R: Rng {
        let only_place = new_tile_boost.roll(rng);
        if self.is_over() {
            return None;
        }
        // Gist of the algorithm: get all moves, legal or otherwise. Shuffle all moves and find the
        // first legal move, i.e., one that does not put into guard.
        let mut moves: Vec<PossibleMove> = self.board.all_valid_moves_ignoring_guard(
            self.current_player_turn,
            WithNewTiles(self.bag_for_current_player().non_empty()),
        ).collect();
        moves.shuffle(rng);
        let moves = moves;
        assert_not!(moves.is_empty());
        Some(
            self.get_random_move_for_current_player_aux(only_place, &moves)
                .or_else(|| self.get_random_move_for_current_player_aux(false, &moves))
                .expect("No moves found")
        )
    }

    fn get_random_move_for_current_player_aux(
        &self, only_place: bool, moves: &Vec<PossibleMove>,
    ) -> Option<PossibleMove> {
        for mv in moves {
            if only_place && (match &mv {
                PossibleMove::PlaceNewTile { .. } => false,
                _ => true,
            }) {
                continue;
            }
            let is_valid = match mv {
                PossibleMove::PlaceNewTile(o, _) => self.is_valid_placement(*o),
                PossibleMove::ApplyNonCommandTileAction { .. } => {
                    self.board.does_not_put_in_guard(
                        self.possible_move_to_board_move(&mv),
                        self.current_player_turn,
                    )
                }
            };
            if is_valid {
                return Some(mv.clone());
            }
        }
        None
    }

    pub fn all_valid_game_moves_for(&self, o: Owner) -> impl Iterator<Item=PossibleMove> + '_ {
        self.board.all_valid_moves(
            o,
            WithNewTiles(self.bag_for_current_player().non_empty()),
        )
    }

    pub fn all_valid_game_moves_for_ignoring_guard(&self, o: Owner) -> impl Iterator<Item=PossibleMove> + '_ {
        self.board.all_valid_moves_ignoring_guard(
            o,
            WithNewTiles(self.bag_for_current_player().non_empty()),
        )
    }

    pub fn as_single_string(&self) -> String {
        single_char_print_state(&self)
    }

    pub fn as_double_string(&self) -> String {
        double_char_print_state(&self)
    }

    pub fn discard_bag_for(&self, o: Owner) -> &DiscardBag {
        match o {
            Owner::TopPlayer => &self.top_player_discard,
            Owner::BottomPlayer => &self.bottom_player_discard,
        }
    }

    pub fn bag_for_current_player(&self) -> &TileBag {
        match self.current_player_turn {
            Owner::TopPlayer => &self.top_player_bag,
            Owner::BottomPlayer => &self.bottom_player_bag,
        }
    }

    pub fn bag_for_other_player(&self) -> &TileBag {
        match self.current_player_turn {
            Owner::TopPlayer => &self.bottom_player_bag,
            Owner::BottomPlayer => &self.top_player_bag,
        }
    }

    pub fn to_undo(&self, mv: &GameMove) -> PossibleMove {
        match mv {
            GameMove::PlaceNewTile(o) => PossibleMove::PlaceNewTile(*o, self.current_player_turn),
            GameMove::PullAndPlay(o) => self.to_undo(&GameMove::PlaceNewTile(*o)),
            GameMove::ApplyNonCommandTileAction { src, dst } => {
                // TODO handle duplication with the other place where capturing is extracted.
                let capturing = self.board.get_board().get(*dst);
                PossibleMove::ApplyNonCommandTileAction {
                    src: *src,
                    dst: *dst,
                    capturing: capturing.cloned(),
                }
            }
        }
    }

    fn pop_moves_stack(&mut self) -> () {
        assert!(self.moves_without_capture_or_placement_stack.len() >= 2);
        self.moves_without_capture_or_placement_stack.pop();
    }
    pub fn undo(&mut self, mv: PossibleMove) -> () {
        self.current_player_turn = self.current_player_turn.next_player();
        match &mv {
            PossibleMove::PlaceNewTile(_, _) => self.pop_moves_stack(),
            PossibleMove::ApplyNonCommandTileAction { capturing, .. } => {
                if capturing.is_some() {
                    self.pop_moves_stack()
                } else {
                    let moves = self.moves_without_capture_or_placement_stack.last_mut().unwrap();
                    assert!(*moves >= 1);
                    *moves -= 1;
                }
            }
        }
        if let Some(t) = self.board.undo(mv) {
            self.pop_moves_stack(); // TODO document why the hell this needs to happen twice o_O.
            let bag = match self.current_player_turn {
                Owner::TopPlayer => &mut self.top_player_bag,
                Owner::BottomPlayer => &mut self.bottom_player_bag,
            };
            assert_eq!(t.owner, self.current_player_turn);
            assert_eq!(t.current_side, CurrentSide::Initial);
            bag.push(t.tile);
        }
    }

    pub fn get(&self, c: Coordinates) -> Option<&PlacedTile> {
        self.board.get(c)
    }
}

impl Hash for GameState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.board.get_board().hash(state)
    }
}

impl Rectangular for GameState {
    fn width(&self) -> u16 {
        self.board.width()
    }

    fn height(&self) -> u16 {
        self.board.height()
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert_empty, assert_eq_set};
    use crate::common::utils::test_rng;
    use crate::game::units;

    use super::*;

    #[test]
    fn get_legal_moves_does_not_allow_the_duke_to_remain_in_guard() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let footman_coordinates = Coordinates { x: 2, y: 2 };
        board.place(footman_coordinates, PlacedTile::new(Owner::TopPlayer, units::footman()));
        board.place(Coordinates { x: 0, y: 1 }, PlacedTile::new(Owner::BottomPlayer, units::footman()));

        assert_empty!(GameState::from_board(board, Owner::TopPlayer).get_legal_moves(footman_coordinates));
    }

    #[test]
    fn get_legal_moves_does_not_allow_the_duke_to_move_into_guard() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, units::duke()));
        board.place(Coordinates { x: 3, y: 1 }, PlacedTile::new(Owner::BottomPlayer, units::footman()));

        assert_eq!(
            vec!(
                (Coordinates { x: 1, y: 0 }, TileAction::Slide),
                (Coordinates { x: 2, y: 0 }, TileAction::Slide),
                (Coordinates { x: 4, y: 0 }, TileAction::Slide),
                (Coordinates { x: 5, y: 0 }, TileAction::Slide),
            ),
            GameState::from_board(board, Owner::TopPlayer).get_legal_moves(duke_coordinates),
        );
    }

    #[test]
    pub fn can_make_a_move_returns_false_if_moving_duke_into_guard() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, units::place_tile(Owner::TopPlayer, units::duke));
        board.place(
            Coordinates { x: 3, y: 1 },
            units::place_tile(Owner::BottomPlayer, units::footman),
        );
        let state = GameState::from_board(board, Owner::TopPlayer);
        assert_not!(state.can_make_a_move(
            &GameMove::ApplyNonCommandTileAction { src: duke_coordinates, dst: Coordinates { x: 3, y: 0 }}));
    }

    #[test]
    fn can_not_pull_from_bag_if_not_tile_removes_duke_from_guard_move_threat() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let bag = TileBag::new(vec!(TileRef::new(units::footman())));
        board.place(
            Coordinates { x: 0, y: 1 },
            PlacedTile::new(Owner::BottomPlayer, units::footman()),
        );

        assert_eq!(
            CanPullNewTileResult::DukeAlwaysInGuard,
            GameState::from_board_with_bag(board, Owner::TopPlayer, bag).can_pull_tile_from_bag(),
        );
    }

    #[test]
    fn can_not_pull_from_bag_if_not_tile_removes_duke_from_guard_jump() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let bag = TileBag::new(vec!(TileRef::new(units::footman())));
        board.place(
            Coordinates { x: 0, y: 2 },
            PlacedTile::new(Owner::BottomPlayer, units::champion()),
        );

        assert_eq!(
            CanPullNewTileResult::DukeAlwaysInGuard,
            GameState::from_board_with_bag(board, Owner::TopPlayer, bag).can_pull_tile_from_bag(),
        );
    }

    #[test]
    fn can_pull_from_bag_returns_correct_value_if_duke_is_in_guard() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let bag = TileBag::new(vec!(TileRef::new(units::footman())));
        let mut opposite_duke = PlacedTile::new(Owner::BottomPlayer, units::duke());
        opposite_duke.flip();
        board.place(Coordinates { x: 0, y: 5 }, opposite_duke);

        let mut state = GameState::from_board_with_bag(board, Owner::TopPlayer, bag);
        assert_eq!( // Place in general is allowed...
                    CanPullNewTileResult::OK,
                    state.can_pull_tile_from_bag(),
        );

        state.pull_tile_from_bag(&mut test_rng());
        assert_not!( // But only below the current duke, since the current duke is in guard.
            state.is_valid_placement(DukeOffset::Right)
        );
        assert!( // Bottom blocks the enemy duke.
                 state.is_valid_placement(DukeOffset::Bottom)
        );
    }

    #[test]
    fn can_place_does_not_allow_the_duke_to_move_into_guard() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let bag = TileBag::new(vec!(TileRef::new(units::footman())));
        let mut opposite_duke = PlacedTile::new(Owner::BottomPlayer, units::duke());
        opposite_duke.flip();
        board.place(Coordinates { x: 0, y: 5 }, opposite_duke);

        let mut state = GameState::from_board_with_bag(board, Owner::TopPlayer, bag);
        assert_eq!( // Place in general is allowed...
                    CanPullNewTileResult::OK,
                    state.can_pull_tile_from_bag(),
        );

        state.pull_tile_from_bag(&mut test_rng());
        assert_not!( // But only below the current duke, since the current duke is in guard.
            state.is_valid_placement(DukeOffset::Right)
        );
        assert!( // Bottom blocks the enemy duke.
                 state.is_valid_placement(DukeOffset::Bottom)
        );
    }

    #[test]
    fn all_valid_game_moves_returns_all_valid_moves() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 2, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let footman_coordinates = Coordinates { x: 5, y: 0 };
        board.place(footman_coordinates, PlacedTile::new(Owner::TopPlayer, units::footman()));
        let bag = TileBag::new(vec!(TileRef::new(units::footman())));
        let mut opposite_duke = PlacedTile::new(Owner::BottomPlayer, units::duke());
        opposite_duke.flip();
        board.place(Coordinates { x: 0, y: 5 }, opposite_duke);

        let state = GameState::from_board_with_bag(board, Owner::TopPlayer, bag);
        assert_eq_set!(
            vec!(
                PossibleMove::PlaceNewTile(DukeOffset::Right, Owner::TopPlayer),
                PossibleMove::PlaceNewTile(DukeOffset::Left, Owner::TopPlayer),
                PossibleMove::PlaceNewTile(DukeOffset::Bottom, Owner::TopPlayer),

                PossibleMove::ApplyNonCommandTileAction {
                    src: duke_coordinates, dst: Coordinates {x: 1, y: 0}, capturing: None},
                PossibleMove::ApplyNonCommandTileAction {
                    src: duke_coordinates, dst: Coordinates {x: 3, y: 0}, capturing: None},
                PossibleMove::ApplyNonCommandTileAction {
                    src: duke_coordinates, dst: Coordinates {x: 4, y: 0}, capturing: None},

                PossibleMove::ApplyNonCommandTileAction {
                    src: footman_coordinates, dst: Coordinates {x: 5, y: 1}, capturing: None},
                PossibleMove::ApplyNonCommandTileAction {
                    src: footman_coordinates, dst: Coordinates {x: 4, y: 0}, capturing: None},
            ),
            state.all_valid_game_moves_for_current_player(),
        )
    }

    fn test_undo_move(mut state: GameState, mv: GameMove) {
        let undo = state.to_undo(&mv);
        let expected = state.clone();
        state.make_a_move(mv, &mut test_rng());
        state.undo(undo);
        assert_eq!(
            expected,
            state,
        )
    }

    #[test]
    fn undo_can_undo_a_pull() {
        test_undo_move(
            GameState::new(
                &TileBag::new(vec!(TileRef::new(units::knight()))),
                (DukeInitialLocation::Right, FootmenSetup::Sides),
                (DukeInitialLocation::Left, FootmenSetup::Right),
            ),
            GameMove::PullAndPlay(DukeOffset::Bottom),
        )
    }

    #[test]
    fn undo_can_undo_a_move_without_capture() {
        test_undo_move(
            GameState::new(
                &TileBag::new(vec!(TileRef::new(units::knight()))),
                (DukeInitialLocation::Right, FootmenSetup::Sides),
                (DukeInitialLocation::Left, FootmenSetup::Right),
            ),
            GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 1, y: 0 },
                dst: Coordinates { x: 1, y: 1 },
            },
        );
    }

    #[test]
    fn undo_can_undo_a_move_with_capture() {
        let mut gs = GameState::new(
            &TileBag::new(vec!(TileRef::new(units::knight()))),
            (DukeInitialLocation::Right, FootmenSetup::Sides),
            (DukeInitialLocation::Left, FootmenSetup::Right),
        );
        let dst = Coordinates { x: 1, y: 1 };
        gs.board.place(dst, PlacedTile::new(Owner::BottomPlayer, units::footman()));
        test_undo_move(
            gs,
            GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 1, y: 0 },
                dst,
            },
        );
    }

    #[test]
    fn undo_can_undo_a_strike_move_with_capture() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let mut pikeman = PlacedTile::new(Owner::TopPlayer, units::pikeman());
        pikeman.flip();
        board.place(Coordinates { x: 1, y: 0 }, pikeman);
        let footman_coordinates = Coordinates { x: 2, y: 2 };
        board.place(footman_coordinates, PlacedTile::new(Owner::BottomPlayer, units::footman()));
        let gs = GameState::from_board(board, Owner::TopPlayer);
        test_undo_move(
            gs,
            GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 1, y: 0 },
                dst: footman_coordinates,
            },
        );
    }

    #[test]
    fn game_result_should_return_ongoing_if_no_winner_nor_tie() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::TopPlayer, units::duke()));
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::BottomPlayer, units::duke()));
        assert_eq!(
            GameState::from_board(board, Owner::TopPlayer).game_result(),
            GameResult::Ongoing,
        );
    }

    #[test]
    fn game_result_should_return_ongoing_if_other_player_still_has_moves() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let mut footman = PlacedTile::new(Owner::TopPlayer, units::footman());
        footman.flip();
        board.place(Coordinates { x: 4, y: 5 }, footman);
        let mut op_duke = PlacedTile::new(Owner::BottomPlayer, units::duke());
        op_duke.flip();
        board.place(Coordinates { x: 5, y: 0 }, op_duke);
        // TopPlayer can still play a footman move
        assert_eq!(
            GameState::from_board(board, Owner::TopPlayer).game_result(),
            GameResult::Ongoing,
        );
    }

    #[test]
    fn game_result_should_return_some_on_winner() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let footman = PlacedTile::new(Owner::TopPlayer, units::footman());
        board.place(Coordinates { x: 4, y: 5 }, footman);
        let mut op_duke = PlacedTile::new(Owner::BottomPlayer, units::duke());
        op_duke.flip();
        board.place(Coordinates { x: 5, y: 0 }, op_duke);
        assert_eq!(
            GameState::from_board(board, Owner::TopPlayer).game_result(),
            GameResult::Won(Owner::BottomPlayer),
        );
    }

    #[test]
    fn game_result_should_return_tie_after_enough_consecutive_moves_with_no_capture_or_placement() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, units::duke()));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, units::duke()));
        let mut gs = GameState::from_board(board, Owner::TopPlayer);
        // TODO base this count on the maximum const
        for _ in 0..7 {
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 0, y: 0 },
                dst: Coordinates { x: 5, y: 0 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 5, y: 5 },
                dst: Coordinates { x: 0, y: 5 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 5, y: 0 },
                dst: Coordinates { x: 5, y: 5 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 0, y: 5 },
                dst: Coordinates { x: 0, y: 0 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 5, y: 5 },
                dst: Coordinates { x: 0, y: 5 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 0, y: 0 },
                dst: Coordinates { x: 5, y: 0 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 0, y: 5 },
                dst: Coordinates { x: 0, y: 0 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 5, y: 0 },
                dst: Coordinates { x: 5, y: 5 },
            }, &mut test_rng());
        }
        assert_eq!(gs.game_result(), GameResult::Tie)
    }

    #[test]
    #[should_panic]
    fn regression_moving_a_duke_in_footman_attack_panics() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));
        board.place(
            Coordinates { x: 2, y: 1 },
            units::place_tile_flipped(Owner::TopPlayer, units::footman),
        );
        let duke_coordinates = Coordinates { x: 1, y: 5 };
        board.place(duke_coordinates, units::place_tile_flipped(Owner::BottomPlayer, units::duke));
        let mut gs = GameState::from_board(board, Owner::BottomPlayer);
        gs.make_a_move(GameMove::ApplyNonCommandTileAction {
            src: duke_coordinates,
            dst: Coordinates { x: 1, y: 2 },
        }, &mut test_rng());
    }
}