use std::borrow::Borrow;
use std::mem;

use crate::common::coordinates::Coordinates;
use crate::game::state::{CanPullNewTileResult, GameState};
use crate::game::state::GameMove;
use crate::view::move_view::MoveView;

#[derive(Debug, Clone)]
pub(super) enum ViewPosition {
    BoardPosition {
        p: Coordinates,
        // If a current tile has been selected for moving, but not yet moved
        moving: Option<Coordinates>,
    },
    Placing(MoveView),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewStateMode {
    FreeMoving(Coordinates),
    MovingSelection(Coordinates, Coordinates),
    Placing(MoveView),
}

pub struct ViewState {
    game_state: GameState,
    view_position: ViewPosition,
    pub info: Option<String>,
}

// In general, mutating methods will panic on an invalid state. Every mutating function also
// supports a bool method to check if the operation is legal.
impl ViewState {
    pub(super) fn get_game_state(&self) -> &GameState {
        &self.game_state
    }
    pub(super) fn get_game_state_mut(&mut self) -> &mut GameState {
        &mut self.game_state
    }
    pub(super) fn get_view_position(&self) -> &ViewPosition {
        &self.view_position
    }

    pub fn current_state(&self) -> ViewStateMode {
        match self.view_position {
            ViewPosition::BoardPosition { p, moving } => moving.map_or(
                ViewStateMode::FreeMoving(p),
                |m| ViewStateMode::MovingSelection(p, m),
            ),
            ViewPosition::Placing(mv) => ViewStateMode::Placing(mv),
        }
    }
    pub fn info<S>(&mut self, str: S) -> () where S: Borrow<str> {
        self.info = Some(str.borrow().to_owned())
    }

    pub fn new(gs: GameState) -> ViewState {
        ViewState {
            game_state: gs,
            view_position: ViewPosition::BoardPosition { p: Coordinates { x: 0, y: 0 }, moving: None },
            info: None,
        }
    }

    pub fn move_view_position(&mut self, mv: MoveView) -> bool {
        match &self.view_position {
            ViewPosition::BoardPosition { p, moving } => {
                match mv.mv(*p, &self.game_state) {
                    Some(c) => {
                        self.view_position = ViewPosition::BoardPosition { p: c, moving: *moving };
                        true
                    }
                    None => false,
                }
            }
            e => panic!("Invalid position for move: {:?}", e)
        }
    }

    pub fn move_placement(&mut self, mv: MoveView) -> bool {
        match &mut self.view_position {
            ViewPosition::Placing(ref mut current_mv) => {
                let result = self.game_state.is_valid_placement(mv.into());
                if result {
                    *current_mv = mv;
                }
                result
            }
            e => panic!("Invalid view position for move placement: {:?}", e),
        }
    }

    pub fn is_moving(&self) -> bool {
        match self.view_position {
            ViewPosition::BoardPosition { p: _, moving } => moving.is_some(),
            _ => false,
        }
    }

    pub fn select_for_movement(&mut self) -> bool {
        match &self.view_position {
            ViewPosition::BoardPosition { p, moving } if moving.is_none() => {
                let tile = self.game_state.get(*p);
                let is_owned_tile =
                    tile.map_or(false, |t| t.owner == self.game_state.current_player_turn());
                let result = is_owned_tile && tile.is_some();
                if result {
                    self.view_position = ViewPosition::BoardPosition { p: *p, moving: Some(*p) };
                }
                result
            }
            _ => panic!("Invalid state for movement selection: {:?}", self.view_position),
        }
    }

    pub fn move_selected(&mut self) -> bool {
        match &self.view_position {
            ViewPosition::BoardPosition { p, moving: Some(m) } => {
                let game_move = GameMove::ApplyNonCommandTileAction { src: *m, dst: *p };
                let result = self.game_state.can_make_a_move(&game_move);
                if result {
                    self.game_state.make_a_move(game_move);
                    self.unselect();
                }
                result
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
                self.game_state.can_pull_tile_from_bag_bool(),
            _ => false,
        }
    }

    pub fn pull_token_from_bag(&mut self) -> () {
        match &self.view_position {
            ViewPosition::BoardPosition { moving: None, .. } => {
                self.game_state.pull_tile_from_bag();
                self.view_position =
                    ViewPosition::Placing(
                        MoveView::relative_direction(
                            self.game_state.current_duke_coordinate(),
                            *self.game_state.empty_spaces_near_current_duke()
                                .first()
                                .expect("No empty space near duke"),
                        ).expect("ASSERTION ERROR: empty space near duke isn't near duke"),
                    );
            }
            e => panic!("Invalid state to pull token: {:?}", e)
        }
    }

    pub fn is_placing(&self) -> bool {
        match &self.view_position {
            ViewPosition::Placing(_) => true,
            _ => false,
        }
    }

    pub(super) fn relative_to_absolute_panicking(&self, c: Coordinates, mv: MoveView) -> Coordinates {
        mv
            .mv(c, &self.game_state)
            .expect(
                format!("ASSERTION ERROR: Invalid duke_offset: {:?}; duke_position: {:?}",
                        &mv,
                        c,
                ).as_str(),
            )
    }

    // place always succeeds (short of panic), since we don't allow the view state to enter an
    // invalid placement state to begin with.
    pub fn place(&mut self) -> () {
        assert!(self.is_placing());
        let p = match &self.view_position {
            ViewPosition::Placing(p) => {
                let duke_coordinate = self.game_state.current_duke_coordinate();
                self.relative_to_absolute_panicking(duke_coordinate, *p)
            }
            e => panic!("Invalid view position for placing: {:?}", e),
        };
        let old = mem::replace(
            &mut self.view_position,
            ViewPosition::BoardPosition { p, moving: None },
        );
        match old {
            ViewPosition::Placing(p) =>
                self.game_state.make_a_move(GameMove::PlaceNewTile(p.into())),
            e => panic!("ASSERTION_ERROR: Invalid view position for placing: {:?}", e),
        };
    }
}
