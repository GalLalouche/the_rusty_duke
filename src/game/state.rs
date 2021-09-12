use crate::assert_not;
use crate::common::coordinates::Coordinates;
use crate::game::board::{DukeInitialLocation, FootmenSetup, GameBoard, GameMove};
use crate::game::tile::{DiscardBag, Owner, Ownership, PlacedTile, Tile, TileAction, TileBag};

pub struct GameState {
    // Shouldn't really be pub, breaks Demeter
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
    OK,
}

impl GameState {
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
            CanPullNewTileResult::OK
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

    // Except commands
    pub fn get_legal_moves(&self, src: Coordinates) -> Vec<(Coordinates, TileAction)> {
        assert!(self.board.get(src).unwrap().owner.same_team(self.current_player_turn));
        self.board.get_legal_moves(src)
    }

    pub fn current_duke_coordinate(&self) -> Coordinates {
        self.board.duke_coordinates(self.current_player_turn)
    }

    pub fn empty_spaces_near_current_duke(&self) -> Vec<Coordinates> {
        self.board.empty_spaces_near_current_duke(self.current_player_turn)
    }
}
