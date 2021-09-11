use tui::buffer::Buffer;
use tui::layout::Rect;
use tui::widgets::{Block, Borders, BorderType, Widget};

use crate::common::coordinates::Coordinates;
use crate::common::utils::Folding;
use crate::game::board::GameBoard;
use crate::game::tile::TileSide;
use crate::view::tui::tile_renderer::{render_board_tile, RenderBoardTileConfig, RenderBoardTileHighlight, RenderTileConfig};

fn tile_width() -> u16 { TileSide::SIDE + 2 }

fn tile_height() -> u16 { TileSide::SIDE + 2 }

fn board_area(area: Rect, board_height: u16, board_width: u16) -> Rect {
    let width = TileSide::SIDE + 2;
    let height = TileSide::SIDE + 2;
    Rect {
        x: area.x,
        y: area.y,
        width: board_width as u16 * width + 2,
        height: board_height as u16 * height + 2,
    }
}

pub struct MovingConfig {
    pub focus: Coordinates,
    pub legal_options: Vec<Coordinates>,
}

pub fn render_board(
    board: &GameBoard,
    hightlighting: Option<Coordinates>,
    moving: Option<MovingConfig>,
    area: Rect,
    buf: &mut Buffer,
) {
    let b = Block::default()
        .title("Game Board")
        .border_type(BorderType::Double)
        .borders(Borders::ALL)
        ;
    let game_board_area = board_area(area, board.height(), board.width());
    let inner_area = b.inner(game_board_area);
    b.render(game_board_area, buf);

    for (c, w) in board.get_board().all_coordinated_values() {
        render_board_tile(
            w,
            RenderTileConfig::Board(
                RenderBoardTileConfig {
                    hovering: hightlighting.has(&c),
                    highlight_style: moving.as_ref().and_then(|e|
                        if e.focus == c {
                            Some(RenderBoardTileHighlight::Moving)
                        } else if e.legal_options.contains(&c) {
                            Some(RenderBoardTileHighlight::LegalMove)
                        } else {
                            None
                        }),
                }
            ),
            Rect {
                x: inner_area.x + tile_width() * c.x as u16,
                y: inner_area.y + tile_height() * c.y as u16,
                width: tile_width(),
                height: tile_height(),
            },
            buf,
        );
    }

    if let Some(tile) = hightlighting.and_then(|c| board.get_board().get(c)) {
        let info_block_position = Rect {
            y: game_board_area.y + game_board_area.height - tile_height() - 1 - 1,
            x: game_board_area.x + game_board_area.width + 1,
            width: tile_width() * 2 + 2,
            height: tile_height() + 2,
        };
        let b = Block::default()
            .title(tile.tile.name.clone() + " info")
            .border_type(BorderType::Double)
            .borders(Borders::ALL)
            ;
        let inner = b.inner(info_block_position);
        b.render(info_block_position, buf);

        let current_position = Rect {
            x: inner.x,
            y: inner.y,
            width: tile_width(),
            height: tile_height(),
        };
        render_board_tile(
            Some(tile),
            RenderTileConfig::Info { title: "Current".to_owned() },
            current_position,
            buf,
        );

        let flipped_position = Rect {
            y: inner.y,
            x: inner.x + tile_width(),
            width: tile_width(),
            height: tile_height(),
        };
        // TODO Yuck!
        let mut flipped_tile = tile.clone();
        flipped_tile.flip();
        render_board_tile(
            Some(&flipped_tile),
            RenderTileConfig::Info { title: "Flipped".to_owned() },
            flipped_position,
            buf,
        );
    }
}