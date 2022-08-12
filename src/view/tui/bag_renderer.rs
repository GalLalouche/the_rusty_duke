use tui::buffer::Buffer;
use tui::layout::Rect;
use tui::widgets::{Block, Borders, BorderType, Widget};

use crate::common::utils::Vectors;
use crate::game::bag::TileBag;
use crate::game::tile::{CurrentSide, Owner};
use crate::view::tui::tile_renderer::{render_tile, RenderTileConfig, TILE_HEIGHT, TILE_WIDTH};

pub fn render_bag(owner: &Owner, bag: &TileBag, area: Rect, buf: &mut Buffer) -> () {
    let b = Block::default()
        .title(format!("{}'s remaining tile bag", owner))
        .border_type(BorderType::Double)
        .borders(Borders::ALL);
    let inner_area = b.inner(area);
    b.render(area, buf);
    let max_tiles_per_row = inner_area.width / TILE_WIDTH;
    for (row, tiles) in bag.remaining().grouped(max_tiles_per_row as usize).into_iter().enumerate() {
        for (column, tile) in tiles.into_iter().enumerate() {
            let mut render_tile_aux = |current_side, y_offset| render_tile(
                tile,
                owner,
                current_side,
                RenderTileConfig::Info {
                    title: format!("{} {:?}", &tile.get_name()[0..3], current_side) },
                Rect {
                    x: inner_area.x + column as u16 * TILE_WIDTH,
                    y: inner_area.y + row as u16 * TILE_HEIGHT * 2 + y_offset,
                    width: TILE_WIDTH,
                    height: TILE_HEIGHT,
                },
                buf,
            );
            render_tile_aux(CurrentSide::Initial, 0);
            render_tile_aux(CurrentSide::Flipped, TILE_HEIGHT);
        }
    }
}
