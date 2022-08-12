use tui::buffer::Buffer;
use tui::layout::Rect;
use tui::widgets::Widget;

use crate::common::coordinates::Coordinates;
use crate::game::state::GameState;
use crate::game::tile::PlacedTile;
use crate::view::controller::Controller;
use crate::view::state::{ViewState, ViewStateMode};
use crate::view::tui::bag_renderer::render_bag;
use crate::view::tui::board_renderer::{MovingConfig, render_board};

impl Widget for &ViewState {
    fn render(self, area: Rect, buf: &mut Buffer) -> () {
        match self.current_state() {
            ViewStateMode::FreeMoving(p) => render_board_aux(
                area, buf, self.get_game_state(), self.info.as_ref(), p, None),
            ViewStateMode::MovingSelection { src, target } => render_board_aux(
                area, buf, self.get_game_state(), self.info.as_ref(), src, Some(target)),
            ViewStateMode::Placing(relative_duke_offset) => {
                let tile = self.get_game_state()
                    .pulled_tile()
                    .clone()
                    .expect("ViewPosition is placing but state has no pulled tile");
                let duke_coordinate = self.get_game_state().current_duke_coordinate();
                let placement =
                    self.relative_to_absolute_panicking(duke_coordinate, relative_duke_offset);
                // Not the most efficient, but safer for now.
                let mut temp_board = self.get_game_state().board().clone();
                temp_board.put(
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
            ViewStateMode::ShowCurrentBag => render_bag(
                &self.get_game_state().current_player_turn(),
                self.get_game_state().bag_for_current_player(),
                area,
                buf,
            ),
            ViewStateMode::ShowOtherBag => render_bag(
                &self.get_game_state().current_player_turn().next_player(),
                self.get_game_state().bag_for_other_player(),
                area,
                buf,
            ),
        }
    }
}

impl Widget for &Controller {
    fn render(self, area: Rect, buf: &mut Buffer) {
        self.get_view_state().render(area, buf);
    }
}

fn render_board_aux(
    area: Rect,
    buf: &mut Buffer,
    gs: &GameState,
    info: Option<&String>,
    p: Coordinates,
    moving: Option<Coordinates>,
) -> () {
    let moving_config =
        if let Some(m) = moving {
            Some(MovingConfig {
                focus: m,
                legal_options: gs
                    .get_legal_moves(m)
                    .iter()
                    .map(|e| e.0)
                    .collect(),
            })
        } else {
            None
        };
    render_board(
        &gs.board(),
        Some(p),
        moving_config,
        info,
        area,
        buf,
    );
}
