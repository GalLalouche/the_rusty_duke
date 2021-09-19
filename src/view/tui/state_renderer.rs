use tui::buffer::Buffer;
use tui::layout::Rect;
use tui::widgets::Widget;

use crate::game::tile::PlacedTile;
use crate::view::controller::Controller;
use crate::view::state::{ViewPosition, ViewState};
use crate::view::tui::board_renderer::{MovingConfig, render_board};

impl Widget for &ViewState {
    fn render(self, area: Rect, buf: &mut Buffer) -> () {
        match self.get_view_position() {
            ViewPosition::BoardPosition { p, moving } => {
                let moving_config =
                    if let Some(m) = moving {
                        Some(MovingConfig {
                            focus: *m,
                            legal_options: self
                                .get_game_state()
                                .get_legal_moves(*m)
                                .iter()
                                .map(|e| e.0)
                                .collect(),
                        })
                    } else {
                        None
                    };
                render_board(
                    &self.get_game_state().board(),
                    Some(*p),
                    moving_config,
                    self.info.as_ref(),
                    area,
                    buf,
                );
            }
            ViewPosition::Placing(relative_duke_offset) => {
                let tile = self.get_game_state()
                    .pulled_tile()
                    .clone()
                    .expect("ViewPosition is placing but state has no pulled tile");
                let duke_coordinate = self.get_game_state().current_duke_coordinate();
                let placement =
                    self.relative_to_absolute_panicking(duke_coordinate, *relative_duke_offset);
                // Not the most efficient, but safer for now.
                let mut temp_board = self.get_game_state().board().clone();
                temp_board.place(
                    placement,
                    PlacedTile::new_from_ref(self.get_game_state().current_player_turn(), tile),
                );
                render_board(
                    &temp_board,
                    Some(placement),
                    Some(MovingConfig {
                        focus: duke_coordinate,
                        legal_options: vec!(placement),
                    }),
                    self.info.as_ref(),
                    area,
                    buf,
                )
            }
        };
    }
}

impl Widget for &Controller {
    fn render(self, area: Rect, buf: &mut Buffer) {
        self.get_view_state().render(area, buf);
    }
}
