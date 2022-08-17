use std::borrow::Borrow;
use std::mem;

use crate::assert_not;
use crate::common::coordinates::Coordinates;
use crate::game::board::PossibleMove;
use crate::game::state::{GameMove, GameResult};
use crate::game::state::GameState;
use crate::view::controller::Error;
use crate::view::move_view::MoveView;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum ViewPosition {
    Basic(Basic),
    Zoomed(Basic, Zoomed),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Basic {
    BoardPosition {
        p: Coordinates,
        // If a current tile has been selected for moving, but not yet moved
        moving: Option<Coordinates>,
    },
    Placing(MoveView),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Zoomed {
    ShowCurrentBag,
    ShowOtherBag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewStateMode {
    FreeMoving(Coordinates),
    MovingSelection { src: Coordinates, target: Coordinates },
    Placing(MoveView),
    ShowCurrentBag,
    ShowOtherBag,
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

    pub fn current_state(&self) -> ViewStateMode {
        match self.view_position {
            ViewPosition::Basic(Basic::BoardPosition { p, moving }) => moving.map_or(
                ViewStateMode::FreeMoving(p),
                |m| ViewStateMode::MovingSelection { src: p, target: m },
            ),
            ViewPosition::Basic(Basic::Placing(mv)) => ViewStateMode::Placing(mv),
            ViewPosition::Zoomed(_, Zoomed::ShowCurrentBag) => ViewStateMode::ShowCurrentBag,
            ViewPosition::Zoomed(_, Zoomed::ShowOtherBag) => ViewStateMode::ShowOtherBag,
        }
    }
    pub fn info<S>(&mut self, str: S) -> () where S: Borrow<str> {
        self.info = Some(str.borrow().to_owned())
    }
    pub fn is_zoomed_in(&self) -> bool {
        match self.view_position {
            ViewPosition::Basic { .. } => false,
            ViewPosition::Zoomed { .. } => true,
        }
    }
    pub fn show_current_owner_bag(&mut self) -> Option<Error> {
        self.show_bag_aux(Zoomed::ShowCurrentBag)
    }
    pub fn show_other_owner_bag(&mut self) -> Option<Error> {
        self.show_bag_aux(Zoomed::ShowOtherBag)
    }
    fn show_bag_aux(&mut self, pos: Zoomed) -> Option<Error> {
        assert_not!(self.is_zoomed_in());
        match self.view_position {
            ViewPosition::Basic(b) => self.view_position = ViewPosition::Zoomed(b, pos),
            _ => panic!(),
        };
        None
    }
    pub fn zoom_out(&mut self) -> Option<Error> {
        assert!(self.is_zoomed_in());
        match self.view_position {
            ViewPosition::Zoomed(old, _) => self.view_position = ViewPosition::Basic(old),
            _ => panic!(),
        }
        None
    }

    pub fn new(gs: GameState) -> ViewState {
        ViewState {
            game_state: gs,
            view_position: ViewPosition::Basic(Basic::BoardPosition {
                p: Coordinates { x: 0, y: 0 },
                moving: None,
            }),
            info: None,
        }
    }

    pub fn move_view_position(&mut self, mv: MoveView) -> bool {
        match &self.view_position {
            ViewPosition::Basic(Basic::BoardPosition { p, moving }) => {
                match mv.mv(*p, &self.game_state) {
                    Some(c) => {
                        self.view_position =
                            ViewPosition::Basic(Basic::BoardPosition { p: c, moving: *moving });
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
            ViewPosition::Basic(Basic::Placing(ref mut current_mv)) => {
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
        match self.current_state() {
            ViewStateMode::MovingSelection { .. } => true,
            _ => false,
        }
    }

    pub fn select_for_movement(&mut self) -> bool {
        match self.current_state() {
            ViewStateMode::FreeMoving(p) => {
                let tile = self.game_state.get(p);
                let is_owned_tile =
                    tile.map_or(false, |t| t.owner == self.game_state.current_player_turn());
                let result = is_owned_tile && tile.is_some();
                if result {
                    self.view_position =
                        ViewPosition::Basic(Basic::BoardPosition { p, moving: Some(p) });
                }
                result
            }
            _ => panic!("Invalid state for movement selection: {:?}", self.view_position),
        }
    }

    pub fn move_selected(&mut self) -> Option<PossibleMove> {
        match self.current_state() {
            ViewStateMode::MovingSelection { src, target } => {
                let game_move = GameMove::ApplyNonCommandTileAction { src: target, dst: src };
                let pm = self.game_state.to_undo(&game_move);
                let result = self.game_state.can_make_a_move(&game_move);
                if result {
                    self.game_state.make_a_move(game_move);
                    self.unselect();
                    Some(pm)
                } else {
                    None
                }
            }
            e => panic!("Invalid state for moving selected: {:?}", e),
        }
    }

    pub fn can_unselect(&self) -> bool { self.is_moving() }
    pub fn unselect(&mut self) -> () {
        assert!(self.can_unselect());
        match &mut self.view_position {
            ViewPosition::Basic(Basic::BoardPosition { ref mut moving, .. }) => *moving = None,
            e => panic!("Invalid view position for unselecting: {:?}", e),
        }
    }

    pub fn can_pull_token_from_bag(&self) -> bool {
        match self.current_state() {
            ViewStateMode::FreeMoving(_) => self.game_state.can_pull_tile_from_bag_bool(),
            _ => false,
        }
    }

    pub fn pull_token_from_bag(&mut self) -> () {
        assert!(self.can_pull_token_from_bag());
        self.game_state.pull_tile_from_bag();
        self.view_position = ViewPosition::Basic(
            Basic::Placing(
                MoveView::relative_direction(
                    self.game_state.current_duke_coordinate(),
                    *self.game_state.empty_spaces_near_current_duke()
                        .first()
                        .expect("No empty space near duke"),
                ).expect("ASSERTION ERROR: empty space near duke isn't near duke"),
            ));
    }

    pub fn is_placing(&self) -> bool {
        match self.current_state() {
            ViewStateMode::Placing { .. } => true,
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
    pub fn place(&mut self) -> PossibleMove {
        assert!(self.is_placing());
        let (p, mv) = match self.current_state() {
            ViewStateMode::Placing(p) => {
                let duke_coordinate = self.game_state.current_duke_coordinate();
                (
                    self.relative_to_absolute_panicking(duke_coordinate, p),
                    PossibleMove::PlaceNewTile(p.into(), self.game_state.current_player_turn()),
                )
            }
            e => panic!("Invalid view position for placing: {:?}", e),
        };
        let old = mem::replace(
            &mut self.view_position,
            ViewPosition::Basic(Basic::BoardPosition { p, moving: None }),
        );
        match old {
            ViewPosition::Basic(Basic::Placing(p)) =>
                self.game_state.make_a_move(GameMove::PlaceNewTile(p.into())),
            e => panic!("ASSERTION_ERROR: Invalid view position for placing: {:?}", e),
        };
        mv
    }

    pub fn undo(&mut self, mv: PossibleMove) {
        self.game_state.undo(mv)
    }

    pub fn game_result(&self) -> GameResult {
        self.game_state.game_result()
    }
}
