use crate::view::state::ViewState;
use crate::view::tui::move_view::MoveView;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ControllerCommand {
    // Movements are applicable when moving around the board/bag/discarded, or when placing a new
    // tile.
    // When placing a new tile, the movement will just select where the tile placement relative to
    // the duke should be. That also means it is idempotent.
    Left, Right, Up, Down,

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
    CannotMove(MoveView),
    CannotSelect,
    CannotEscape,
    CannotPull,
}

pub struct Controller {
    state: ViewState,
}

impl Controller {
    pub fn new(state: ViewState) -> Controller {
        Controller { state }
    }
    pub(super) fn get_view_state(&self) -> &ViewState {
        &self.state
    }
    fn mv(&mut self, mv: MoveView) -> Option<Error> {
        if self.state.can_move_placement(mv) {
            self.state.move_placement(mv);
            None
        } else if self.state.can_move_view_position(mv) {
            self.state.move_view_position(mv);
            None
        } else {
            Some(Error::CannotMove(mv))
        }
    }
    pub fn apply(&mut self, cm: ControllerCommand) -> Option<Error> {
        match cm {
            ControllerCommand::Left => self.mv(MoveView::Left),
            ControllerCommand::Right => self.mv(MoveView::Right),
            ControllerCommand::Up => self.mv(MoveView::Up),
            ControllerCommand::Down => self.mv(MoveView::Down),
            ControllerCommand::Select =>
            // TODO extract common pattern (using lazy evaluation?)
                if self.state.is_moving() {
                    self.state.move_selected();
                    None
                } else if self.state.is_placing() {
                    self.state.place();
                    None
                } else if self.state.can_select_for_movement() {
                    self.state.select_for_movement();
                    None
                } else {
                    Some(Error::CannotSelect)
                }
            ControllerCommand::PullFromBag =>
                if self.state.can_pull_token_from_bag() {
                    self.state.pull_token_from_bag();
                    None
                } else {
                    Some(Error::CannotPull)
                }

            ControllerCommand::Escape => {
                if self.state.can_unselect() {
                    self.state.unselect();
                    None
                } else {
                    Some(Error::CannotEscape)
                }
            }
        }
    }
}
