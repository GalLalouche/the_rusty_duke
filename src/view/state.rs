use std::mem;

use crate::common::coordinates::Coordinates;
use crate::common::utils::Folding;
use crate::game::board::GameMove;
use crate::game::state::{CanPullNewTileResult, GameState};
use crate::game::tile::Tile;
use crate::view::tui::move_view::MoveView;

#[derive(Debug, Clone)]
pub(super) enum ViewPosition {
    BoardPosition {
        p: Coordinates,
        // If a current tile has been selected for moving, but not yet moved
        moving: Option<Coordinates>,
    },
    Placing(MoveView, Tile),
}

pub struct ViewState {
    game_state: GameState,
    view_position: ViewPosition,
}

// In general, mutating methods will panic on an invalid state. Every mutating function also
// supports a bool method to check if the operation is legal.
impl ViewState {
    pub(super) fn get_game_state(&self) -> &GameState {
        &self.game_state
    }
    pub(super) fn get_view_position(&self) -> &ViewPosition {
        &self.view_position
    }

    pub fn new(gs: GameState) -> ViewState {
        ViewState {
            game_state: gs,
            view_position: ViewPosition::BoardPosition { p: Coordinates { x: 0, y: 0 }, moving: None },
        }
    }

    pub fn can_move_view_position(&self, mv: MoveView) -> bool {
        match self.view_position {
            ViewPosition::BoardPosition { p, .. } =>
                mv.mv(p, self.game_state.board.get_board()).is_some(),
            _ => false,
        }
    }
    pub fn move_view_position(&mut self, mv: MoveView) -> () {
        match &self.view_position {
            ViewPosition::BoardPosition { p, moving } =>
                match mv.mv(*p, self.game_state.board.get_board()) {
                    Some(c) => self.view_position = ViewPosition::BoardPosition { p: c, moving: *moving },
                    None => panic!("Cannot move from {:?} to {:?}", p, mv),
                },
            e => panic!("Invalid position for move: {:?}", e)
        }
    }

    pub fn can_move_placement(&self, mv: MoveView) -> bool {
        let duke_coordinate = self.game_state.current_duke_coordinate();
        self.is_placing() && mv
            .mv(duke_coordinate, &self.game_state.board.get_board())
            .exists(|c| self.game_state.board.get_board().is_empty(*c))
    }
    pub fn move_placement(&mut self, mv: MoveView) -> () {
        assert!(self.can_move_placement(mv));
        match &mut self.view_position {
            ViewPosition::Placing(ref mut current_mv, ..) => *current_mv = mv,
            e => panic!("Invalid view position for move placement: {:?}", e),
        }
    }

    pub fn is_moving(&self) -> bool {
        match self.view_position {
            ViewPosition::BoardPosition { p: _, moving } => moving.is_some(),
            _ => false,
        }
    }

    fn select_for_movement_aux(&self) -> Option<ViewPosition> {
        match &self.view_position {
            ViewPosition::BoardPosition { p, moving } if moving.is_none() => {
                let tile = self.game_state.board.get(*p);
                let is_owned_tile =
                    tile.map_or(false, |t| t.owner == self.game_state.current_player_turn);
                let result = is_owned_tile && tile.is_some();
                if result {
                    Some(ViewPosition::BoardPosition { p: *p, moving: Some(*p) })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn can_select_for_movement(&self) -> bool {
        self.select_for_movement_aux().is_some()
    }
    pub fn select_for_movement(&mut self) -> () {
        match self.select_for_movement_aux() {
            None => panic!("Invalid state for movement selection: {:?}", self.view_position),
            Some(vp) => self.view_position = vp,
        }
    }

    pub fn move_selected(&mut self) -> () {
        match &self.view_position {
            ViewPosition::BoardPosition { p, moving: Some(m) } => {
                let game_move = GameMove::ApplyNonCommandTileAction { src: *m, dst: *p };
                if self.game_state.can_make_a_move(&game_move) {
                    self.game_state.make_a_move(game_move);
                    self.unselect();
                } else {
                    panic!("Cannot move selected: {:?}", self.view_position)
                }
            }
            e => panic!("Invalid state for moving selected: {:?}", e),
        }
    }

    pub fn can_unselect(&self) -> bool {
        match self.view_position {
            ViewPosition::BoardPosition { moving: Some(_), .. } => true,
            _ => false,
        }
    }
    pub fn unselect(&mut self) -> () {
        assert!(self.can_unselect());
        match &mut self.view_position {
            ViewPosition::BoardPosition { ref mut moving, .. } => *moving = None,
            e => panic!("Invalid view position for unselecting: {:?}", e),
        }
    }

    pub fn can_pull_token_from_bag(&self) -> bool {
        match &self.view_position {
            ViewPosition::BoardPosition { moving: None, .. } =>
                self.game_state.can_pull_tile_from_bag() == CanPullNewTileResult::OK,
            _ => false,
        }
    }

    pub fn pull_token_from_bag(&mut self) -> () {
        match &self.view_position {
            ViewPosition::BoardPosition { moving: None, .. } =>
                self.view_position =
                    ViewPosition::Placing(
                        MoveView::relative_direction(
                            self.game_state.current_duke_coordinate(),
                            *self.game_state.empty_spaces_near_current_duke()
                                .first()
                                .expect("No empty space near duke"),
                        ).expect("ASSERTION ERROR: empty space near duke isn't near duke"),
                        self.game_state.pull_tile_from_bag(),
                    ),
            e => panic!("Invalid state to pull token: {:?}", e)
        }
    }

    pub fn is_placing(&self) -> bool {
        match &self.view_position {
            ViewPosition::Placing(_, _) => true,
            _ => false,
        }
    }

    pub(super) fn relative_to_absolute_panicking(&self, c: Coordinates, mv: MoveView) -> Coordinates {
        mv
            .mv(c, &self.game_state.board.get_board())
            .expect(
                format!("ASSERTION ERROR: Invalid duke_offset: {:?}; duke_position: {:?}",
                        &mv,
                        c,
                ).as_str(),
            )
    }

    pub fn place(&mut self) -> () {
        assert!(self.is_placing());
        let p = match &self.view_position {
            ViewPosition::Placing(p, _) => {
                let duke_coordinate = self.game_state.current_duke_coordinate();
                self.relative_to_absolute_panicking(duke_coordinate, *p)
            }
            e => panic!("ASSERTION_ERROR: Invalid view position for placing: {:?}", e),
        };
        let old = mem::replace(
            &mut self.view_position,
            ViewPosition::BoardPosition { p, moving: None },
        );
        match old {
            ViewPosition::Placing(p, tile) =>
                self.game_state.make_a_move(GameMove::PlaceNewTile(tile, p.into())),
            e => panic!("ASSERTION_ERROR: Invalid view position for placing: {:?}", e),
        }
    }
}
