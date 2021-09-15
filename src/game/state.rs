use crate::assert_not;
use crate::common::coordinates::Coordinates;
use crate::game::board::{DukeInitialLocation, DukeOffset, FootmenSetup, GameBoard, GameMove};
use crate::game::tile::{DiscardBag, Owner, Ownership, PlacedTile, Tile, TileAction, TileBag};
use crate::game::units;
use strum::IntoEnumIterator;

#[derive(Debug, Clone)]
pub struct GameState {
    // Shouldn't really be pub, breaks Demeter. At the very not be mutable...
    pub board: GameBoard,
    pub current_player_turn: Owner,
    pub player_1_bag: TileBag,
    pub player_1_discard: DiscardBag,
    pub player_2_bag: TileBag,
    pub player_2_discard: DiscardBag,

    is_waiting_for_tile_placement: bool,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CanPullNewTileResult {
    EmptyBag,
    NoSpaceNearDuke,
    DukeAlwaysInGuard,
    OK,
}

impl GameState {
    #[cfg(test)]
    fn from_board(board: GameBoard) -> GameState {
        GameState::from_board_with_bag(board, TileBag::empty())
    }
    #[cfg(test)]
    fn from_board_with_bag(board: GameBoard, bag: TileBag) -> GameState {
        GameState {
            board,
            current_player_turn: Owner::TopPlayer,
            player_1_bag: bag.clone(),
            player_1_discard: DiscardBag::empty(),
            player_2_bag: bag,
            player_2_discard: DiscardBag::empty(),
            is_waiting_for_tile_placement: false,
        }
    }
    pub fn new(
        base_bag: &TileBag,
        player_1_setup: (DukeInitialLocation, FootmenSetup),
        player_2_setup: (DukeInitialLocation, FootmenSetup),
    ) -> GameState {
        let board = GameBoard::setup(player_1_setup, player_2_setup);

        GameState {
            board,
            current_player_turn: Owner::TopPlayer,
            player_1_bag: base_bag.clone(),
            player_1_discard: DiscardBag::empty(),
            player_2_bag: base_bag.clone(),
            player_2_discard: DiscardBag::empty(),
            is_waiting_for_tile_placement: false,
        }
    }

    pub fn rows(&self) -> &Vec<Vec<Option<PlacedTile>>> {
        self.board.rows()
    }

    pub fn can_pull_tile_from_bag(&self) -> CanPullNewTileResult {
        let bag = match self.current_player_turn {
            Owner::TopPlayer => &self.player_1_bag,
            Owner::BottomPlayer => &self.player_2_bag,
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

    pub fn pull_tile_from_bag(&mut self) -> Tile {
        let can_pull = self.can_pull_tile_from_bag();
        assert_eq!(
            can_pull, CanPullNewTileResult::OK, "Cannot pull new tile from bag: {:?}", can_pull);
        let bag = match self.current_player_turn {
            Owner::TopPlayer => &mut self.player_1_bag,
            Owner::BottomPlayer => &mut self.player_2_bag,
        };
        let result =
            bag.pull().expect("Assertion Error: bag should not have been empty by this point");
        self.is_waiting_for_tile_placement = true;
        result
    }

    pub fn can_make_a_move(&mut self, game_move: &GameMove) -> bool {
        match game_move {
            GameMove::PlaceNewTile(_, _) => self.is_waiting_for_tile_placement,
            GameMove::ApplyNonCommandTileAction { src, dst } => self.board.can_move(*src, *dst),
            GameMove::CommandAnotherTile { .. } => unimplemented!(),
        }
    }
    pub fn make_a_move(&mut self, game_move: GameMove) -> () {
        if let GameMove::PlaceNewTile(_, _) = game_move {
            assert!(self.is_waiting_for_tile_placement, "Invalid state for placing a new tile");
        } else {
            assert_not!(self.is_waiting_for_tile_placement, "Waiting for a new tile placement");
        }
        self.board.make_a_move(game_move, self.current_player_turn);
        self.current_player_turn = self.current_player_turn.next_player();
        if self.is_waiting_for_tile_placement {
            self.is_waiting_for_tile_placement = false;
        }
    }

    pub fn get_tiles_for_current_owner(&self) -> Vec<(Coordinates, &PlacedTile)> {
        self.board.get_tiles_for(self.current_player_turn)
    }

    fn does_not_put_in_guard(&self, gm: GameMove) -> bool {
        let mut state_clone = self.clone();
        state_clone.board.make_a_move(gm, self.current_player_turn);
        !state_clone.is_guard()
    }
    // Except commands
    pub fn get_legal_moves(&self, src: Coordinates) -> Vec<(Coordinates, TileAction)> {
        assert!(self.board.get(src).unwrap().owner.same_team(self.current_player_turn));
        self.board
            .get_legal_moves(src)
            .into_iter()
            .filter(|(dst, _)|
                self.does_not_put_in_guard(GameMove::ApplyNonCommandTileAction { src, dst: *dst }))
            .collect()
    }

    pub fn current_duke_coordinate(&self) -> Coordinates {
        self.board.duke_coordinates(self.current_player_turn)
    }

    pub fn empty_spaces_near_current_duke(&self) -> Vec<Coordinates> {
        self.board.empty_spaces_near_current_duke(self.current_player_turn)
    }

    pub fn is_valid_placement(&self, offset: DukeOffset) -> bool {
        self.board.is_valid_placement(self.current_player_turn, offset) &&
            self.does_not_put_in_guard(GameMove::PlaceNewTile(units::footman(), offset))
    }
    // TODO use this to verify the duke is not in guard after a move
    pub fn is_guard(&self) -> bool {
        self.board.is_attacked(self.current_duke_coordinate(), self.current_player_turn)
    }
    // pub fn is_over(&self) -> bool {
    //     // TODO handle "stalemate" (which is a loss in Duke).
    //     let current_duke = self.current_duke_coordinate();
    //     let current_player = self.current_player_turn;
    //     !self.board.has_legal_moves()
    //     let is_defended = |c| -> bool {
    //         self.board
    //             .get_board()
    //             .active_coordinates()
    //             .iter()
    //             .filter(|e| e.1.owner.different_team(current_player))
    //             .any(|defender|
    //                 self
    //                     .get_legal_moves(defender.0)
    //                     .iter()
    //                     .any(|defender_move| defender_move.0 == c)
    //             )
    //     };
    //     is_defended(current_duke) &&
    //         self.get_legal_moves(current_duke).into_iter().map(|e| e.0).all(|c| is_defended(c))
    // }

    // TODO validate legal duke moves
}

#[cfg(test)]
mod tests {
    use crate::assert_empty;
    use crate::game::units;

    use super::*;

    #[test]
    fn get_legal_moves_does_not_allow_the_duke_to_remain_in_guard() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let footman_coordinates = Coordinates { x: 2, y: 2 };
        board.place(footman_coordinates, PlacedTile::new(Owner::TopPlayer, units::footman()));
        board.place(Coordinates { x: 0, y: 1 }, PlacedTile::new(Owner::BottomPlayer, units::footman()));

        assert_empty!(GameState::from_board(board).get_legal_moves(footman_coordinates));
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
            GameState::from_board(board).get_legal_moves(duke_coordinates),
        );
    }

    #[test]
    fn can_not_pull_from_bag_if_not_tile_removes_duke_from_guard_move_threat() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let bag = TileBag::new(vec!(units::footman()));
        board.place(
            Coordinates { x: 0, y: 1 },
            PlacedTile::new(Owner::BottomPlayer, units::footman()),
        );

        assert_eq!(
            CanPullNewTileResult::DukeAlwaysInGuard,
            GameState::from_board_with_bag(board, bag).can_pull_tile_from_bag(),
        );
    }

    #[test]
    fn can_not_pull_from_bag_if_not_tile_removes_duke_from_guard_jump() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let bag = TileBag::new(vec!(units::footman()));
        board.place(
            Coordinates { x: 0, y: 2 },
            PlacedTile::new(Owner::BottomPlayer, units::champion()),
        );

        assert_eq!(
            CanPullNewTileResult::DukeAlwaysInGuard,
            GameState::from_board_with_bag(board, bag).can_pull_tile_from_bag(),
        );
    }

    #[test]
    fn can_pull_from_bag_returns_correct_value_if_duke_is_in_guard() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let bag = TileBag::new(vec!(units::footman()));
        let mut opposite_duke = PlacedTile::new(Owner::BottomPlayer, units::duke());
        opposite_duke.flip();
        board.place(Coordinates { x: 0, y: 5 }, opposite_duke);

        let mut state = GameState::from_board_with_bag(board, bag);
        assert_eq!( // Place in general is allowed...
                    CanPullNewTileResult::OK,
                    state.can_pull_tile_from_bag(),
        );

        state.pull_tile_from_bag();
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
        let bag = TileBag::new(vec!(units::footman()));
        let mut opposite_duke = PlacedTile::new(Owner::BottomPlayer, units::duke());
        opposite_duke.flip();
        board.place(Coordinates { x: 0, y: 5 }, opposite_duke);

        let mut state = GameState::from_board_with_bag(board, bag);
        assert_eq!( // Place in general is allowed...
                    CanPullNewTileResult::OK,
                    state.can_pull_tile_from_bag(),
        );

        state.pull_tile_from_bag();
        assert_not!( // But only below the current duke, since the current duke is in guard.
            state.is_valid_placement(DukeOffset::Right)
        );
        assert!( // Bottom blocks the enemy duke.
                 state.is_valid_placement(DukeOffset::Bottom)
        );
    }
}