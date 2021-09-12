use tui::buffer::Buffer;
use tui::layout::Rect;
use tui::widgets::Widget;

use crate::game::tile::PlacedTile;
use crate::view::state::{ViewPosition, ViewState};
use crate::view::tui::board_renderer::{MovingConfig, render_board};
use crate::view::controller::Controller;

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
                render_board(&self.get_game_state().board, Some(*p), moving_config, area, buf);
            }
            ViewPosition::Placing(relative_duke_offset, tile) => {
                let duke_coordinate = self.get_game_state().current_duke_coordinate();
                let placement =
                    self.relative_to_absolute_panicking(duke_coordinate, *relative_duke_offset);
                // Not the most efficient, but safer for now.
                let mut temp_board = self.get_game_state().board.clone();
                temp_board.place(
                    placement,
                    PlacedTile::new(self.get_game_state().current_player_turn, tile.clone()),
                );
                render_board(
                    &temp_board,
                    Some(placement),
                    Some(MovingConfig {
                        focus: duke_coordinate,
                        legal_options: vec!(placement),
                    }),
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
