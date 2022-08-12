use std::borrow::Borrow;

use tui::buffer::Buffer;
use tui::layout::Rect;
use tui::widgets::{Block, Borders, BorderType, Paragraph, Widget, Wrap};

use crate::common::board::Board;
use crate::common::coordinates::Coordinates;
use crate::common::geometry::Rectangular;
use crate::common::utils::Folding;
use crate::game::tile::PlacedTile;
use crate::game::tile_side::TileSide;
use crate::view::tui::tile_renderer::{render_board_tile, RenderBoardTileConfig, RenderBoardTileHighlight, RenderTileConfig, TILE_HEIGHT, TILE_WIDTH};

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
    board: &Board<PlacedTile>,
    hightlighting: Option<Coordinates>,
    moving: Option<MovingConfig>,
    info: Option<&String>,
    area: Rect,
    buf: &mut Buffer,
) where {
    let b = Block::default()
        .title("Game Board")
        .border_type(BorderType::Double)
        .borders(Borders::ALL)
        ;
    let game_board_area = board_area(area, board.height(), board.width());
    let inner_area = b.inner(game_board_area);
    b.render(game_board_area, buf);

    for (c, w) in board.all_coordinated_values() {
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
                x: inner_area.x + TILE_WIDTH * c.x as u16,
                y: inner_area.y + TILE_HEIGHT * c.y as u16,
                width: TILE_WIDTH,
                height: TILE_HEIGHT,
            },
            buf,
        );
    }

    if let Some(tile) = hightlighting.and_then(|c| board.get(c)) {
        let tile_info_block_position = Rect {
            y: game_board_area.y + game_board_area.height - TILE_HEIGHT - 1 - 1,
            x: game_board_area.x + game_board_area.width + 1,
            width: TILE_WIDTH * 2 + 2,
            height: TILE_HEIGHT + 2,
        };
        let b = Block::default()
            .title(tile.tile.get_name().clone() + " info")
            .border_type(BorderType::Double)
            .borders(Borders::ALL)
            ;
        let inner = b.inner(tile_info_block_position);
        b.render(tile_info_block_position, buf);

        let current_position = Rect {
            x: inner.x,
            y: inner.y,
            width: TILE_WIDTH,
            height: TILE_HEIGHT,
        };
        render_board_tile(
            Some(tile),
            RenderTileConfig::Info { title: "Current".to_owned() },
            current_position,
            buf,
        );

        let flipped_position = Rect {
            y: inner.y,
            x: inner.x + TILE_WIDTH,
            width: TILE_WIDTH,
            height: TILE_HEIGHT,
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

    if let Some(s) = info {
        let info_message_area = Rect {
            y: game_board_area.height + 2,
            x: 1,
            width: game_board_area.width + 2 + 20,
            height: 5,
        };
        let b = Block::default()
            .title("Info messages")
            .borders(Borders::ALL)
            ;
        let p = Paragraph::new(s.borrow())
            .block(b)
            .wrap(Wrap { trim: true })
            ;

        p.render(info_message_area, buf)
    }
}