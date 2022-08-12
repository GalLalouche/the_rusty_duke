use std::borrow::BorrowMut;

use rand::thread_rng;

use crate::common::coordinates::Coordinates;
use crate::game::ai::player::ArtificialPlayer;
use crate::game::tile::Owner;
use crate::view::controller::Error::*;
use crate::view::move_view::MoveView;
use crate::view::state::{ViewState, ViewStateMode};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ControllerCommand {
    // Movements are applicable when moving around the board/bag/discarded, or when placing a new
    // tile.
    // When placing a new tile, the movement will just select where the tile placement relative to
    // the duke should be. That also means it is idempotent.
    Left,
    Right,
    Up,
    Down,

    // When moving around the board, selecting a friendly unit will mark it for movement.
    // When a friendly unit is selected, selected a valid target will move it there (or apply its
    // action in the case of command/strike).
    // When trying to place a unit, selecting a space near the duke will place it.
    Select,

    // TODO figure out a way to pass the seed.
    // Will pull random tile from when possible.
    PullFromBag,

    // When trying to move a unit, escape will cancel the selection.
    Escape,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Error {
    MoveOutOfBoard(MoveView),
    CannotSelect(Coordinates),
    CannotPlace(MoveView),
    CannotMove(Coordinates, Coordinates),

    CannotPullFromEmptyBag,
    CannotPullNoRoom,

    CannotEscapeNoSelection,
    CannotEscapeFromPlacing,

    CannotMoveInGuard,
    DoesNotMoveOutOfGuard,

    // Catch-all error for when a specific command is not possible due to the current state.
    InvalidCommand,
}

impl Error {
    pub fn error_msg(&self) -> String {
        match self {
            MoveOutOfBoard(mv) => format!("Cannot move view position to {:?} as it is out of board boundaries", mv),
            CannotPlace(mv) => format!("Cannot place in offset {:?}", mv),
            CannotSelect(c) => format!("Cannot select in position {:?}", c),
            CannotMove(c1, c2) => format!("Cannot move tile in {:?} to {:?}", c1, c2),
            CannotPullFromEmptyBag => "Cannot pull from an empty bag".to_owned(),
            CannotPullNoRoom => "No room to pull into".to_owned(),
            CannotEscapeNoSelection => "Nothing is selected".to_owned(),
            CannotEscapeFromPlacing => "Cannot escape from placing, cheater!".to_owned(),
            CannotMoveInGuard => "Cannot move to a position that would put your duke in guard".to_owned(),
            DoesNotMoveOutOfGuard => "Your duke is still under guard".to_owned(),
            InvalidCommand => "Invalid command".to_owned(),
        }
    }
}

pub struct Controller {
    state: ViewState,
}

impl Controller {
    pub fn current_player_turn(&self) -> Owner {
        self.state.get_game_state().current_player_turn()
    }
    pub fn add_info(&mut self, str: &str) -> () {
        self.state.info(str)
    }
    pub fn new(state: ViewState) -> Controller {
        Controller { state }
    }
    pub(super) fn get_view_state(&self) -> &ViewState {
        &self.state
    }
    pub fn apply(&mut self, cm: ControllerCommand) -> Option<Error> {
        match cm {
            ControllerCommand::Left => self.mv(MoveView::Left),
            ControllerCommand::Right => self.mv(MoveView::Right),
            ControllerCommand::Up => self.mv(MoveView::Up),
            ControllerCommand::Down => self.mv(MoveView::Down),
            ControllerCommand::Select => self.select(),
            ControllerCommand::PullFromBag =>
                if self.state.can_pull_token_from_bag() {
                    self.state.pull_token_from_bag();
                    None
                } else {
                    Some(InvalidCommand)
                }

            ControllerCommand::Escape => {
                if self.state.can_unselect() {
                    self.state.unselect();
                    None
                } else {
                    Some(InvalidCommand)
                }
            }
        }
    }

    fn mv(&mut self, mv: MoveView) -> Option<Error> {
        match self.state.current_state() {
            ViewStateMode::FreeMoving(_) =>
                if self.state.move_view_position(mv) {
                    None
                } else {
                    Some(Error::MoveOutOfBoard(mv))
                },
            ViewStateMode::MovingSelection(_, _) =>
                if self.state.move_view_position(mv) {
                    None
                } else {
                    Some(Error::MoveOutOfBoard(mv))
                },
            ViewStateMode::Placing(_) =>
                if self.state.move_placement(mv) {
                    None
                } else {
                    Some(Error::CannotPlace(mv))
                },
        }
    }

    fn select(&mut self) -> Option<Error> {
        match self.state.current_state() {
            ViewStateMode::FreeMoving(c) =>
                if self.state.select_for_movement() {
                    None
                } else {
                    Some(Error::CannotSelect(c))
                }
            ViewStateMode::MovingSelection(c1, c2) =>
                if self.state.move_selected() {
                    None
                } else {
                    Some(Error::CannotMove(c1, c2))
                }
            ViewStateMode::Placing(_) => {
                self.state.place();
                None
            }
        }
    }

    pub fn ai_move<AI>(&mut self, ai: &AI) where AI: ArtificialPlayer {
        let mut rng = thread_rng();
        ai.play_next_move(rng.borrow_mut(), self.state.get_game_state_mut());
    }

    pub fn is_over(&self) -> Option<Owner> {
        self.state.winner()
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_none;
    use crate::assert_some;
    use crate::game::bag::TileBag;
    use crate::game::board_setup::{DukeInitialLocation, FootmenSetup};
    use crate::game::state::GameState;
    use crate::game::tile::TileRef;
    use crate::game::units;

    use super::*;

    fn setup() -> Controller {
        Controller::new(ViewState::new(GameState::new(
            &TileBag::new(vec!(
                TileRef::new(units::footman()),
                TileRef::new(units::footman()),
                TileRef::new(units::pikeman()),
                TileRef::new(units::pikeman()),
            )),
            (DukeInitialLocation::Left, FootmenSetup::Right),
            (DukeInitialLocation::Right, FootmenSetup::Right),
        )))
    }

    fn check_commands(commands: Vec<ControllerCommand>) -> () {
        let mut controller = setup();
        for command in commands {
            assert_none!(controller.apply(command));
        }
    }

    fn check_commands_for(controller: &mut Controller, commands: Vec<ControllerCommand>) -> () {
        for command in commands {
            assert_none!(controller.apply(command));
        }
    }

    #[test]
    fn can_move_around() {
        check_commands(vec!(
            ControllerCommand::Right,
            ControllerCommand::Down,
            ControllerCommand::Left,
            ControllerCommand::Up,
        ));
    }

    #[test]
    fn can_pull_and_place() {
        check_commands(vec!(
            ControllerCommand::PullFromBag,
            ControllerCommand::Select,
        ));
    }

    #[test]
    fn can_select_and_move() {
        check_commands(vec!(
            ControllerCommand::Right,
            ControllerCommand::Right,
            ControllerCommand::Select,
            ControllerCommand::Down,
            ControllerCommand::Select,
        ));
    }

    #[test]
    fn cannot_move_into_guard() {
        let mut controller = Controller::new(ViewState::new(GameState::new(
            &TileBag::new(Vec::new()),
            (DukeInitialLocation::Left, FootmenSetup::Left),
            (DukeInitialLocation::Right, FootmenSetup::Right),
        )));
        // Move first duke.
        check_commands_for(&mut controller, vec!(
            ControllerCommand::Right,
            ControllerCommand::Right,
            ControllerCommand::Right,
            ControllerCommand::Select,
            ControllerCommand::Left,
            ControllerCommand::Select,
        ));

        // Select second duke.
        check_commands_for(&mut controller, vec!(
            ControllerCommand::Down,
            ControllerCommand::Down,
            ControllerCommand::Down,
            ControllerCommand::Down,
            ControllerCommand::Down,
            ControllerCommand::Right,
            ControllerCommand::Select,
            ControllerCommand::Left,
        ));

        // Try to move second duke will fail
        assert_some!(
            Error::CannotMove(Coordinates {x: 2, y: 5}, Coordinates{x: 3, y: 5}),
            controller.apply(ControllerCommand::Select),
        )
    }
}