use tui::buffer::Buffer;
use tui::layout::Rect;
use tui::style::{Color, Style};
use tui::widgets::{Block, Borders, BorderType, Widget};

use crate::common::coordinates::Coordinates;
use crate::game::offset::{HorizontalOffset, Offsets, VerticalOffset};
use crate::game::tile::{CurrentSide, Owner, PlacedTile, Tile};
use crate::game::tile::Owner::TopPlayer;
use crate::game::tile_side::{TileAction, TileSide};

pub const TILE_WIDTH: u16 = TileSide::SIDE + 2;
pub const TILE_HEIGHT: u16 = TileSide::SIDE + 2;

pub(super) enum RenderBoardTileHighlight {
    Moving,
    LegalMove,
}

pub(super) struct RenderBoardTileConfig {
    pub hovering: bool,
    pub highlight_style: Option<RenderBoardTileHighlight>,
}

pub(super) enum RenderTileConfig {
    Board(RenderBoardTileConfig),
    Info { title: String },
}

pub(super) fn render_board_tile(
    o: Option<&PlacedTile>,
    config: RenderTileConfig,
    area: Rect,
    buf: &mut Buffer,
) -> () {
    match o {
        None => block(&config).render(area, buf),
        Some(o) => render_tile(
            &o.tile,
            &o.owner,
            o.current_side,
            config,
            area,
            buf,
        ),
    }
}

pub(super) fn render_tile(
    tile: &Tile,
    owner: &Owner,
    current_side: CurrentSide,
    config: RenderTileConfig,
    area: Rect,
    buf: &mut Buffer,
) -> () {
    let b = block(&config);
    let with_title = {
        let color = match owner {
            TopPlayer => Color::LightCyan,
            Owner::BottomPlayer => Color::LightMagenta,
        };
        b.title(match &config {
            RenderTileConfig::Info { title } => title.to_owned(),
            _ => tile.get_name().clone(),
        }).style(Style::default().bg(color))
    };
    let inner = with_title.inner(area);
    with_title.render(area, buf);
    let side = match current_side {
        CurrentSide::Initial => tile.get_side_a(),
        CurrentSide::Flipped => tile.get_side_b(),
    };
    for (c, t) in side.board().all_coordinated_values() {
        let normalized_c: Coordinates = c.into();
        buf.set_string(
            inner.x + normalized_c.x as u16,
            inner.y + normalized_c.y as u16,
            to_char(c, t.cloned(), current_side).to_string(), Style::default(),
        );
    }
}

fn block(config: &RenderTileConfig) -> Block {
    let highlight = match &config {
        RenderTileConfig::Board(RenderBoardTileConfig { hovering, .. }) => *hovering,
        _ => false,
    };
    let color = match &config {
        RenderTileConfig::Board(RenderBoardTileConfig { highlight_style: Some(t), .. }) =>
            match t {
                RenderBoardTileHighlight::Moving => Color::LightBlue,
                RenderBoardTileHighlight::LegalMove => Color::LightGreen,
            },
        _ => Color::White,
    };
    Block::default()
        .borders(Borders::ALL)
        .border_type(if highlight { BorderType::Thick } else { BorderType::Plain })
        .border_style(Style::default().fg(color))
}

fn to_char(c: Coordinates, t: Option<TileAction>, side: CurrentSide) -> char {
    match t {
        Some(TileAction::Move) => '●',
        Some(TileAction::Jump) => '○',
        Some(TileAction::Slide) => {
            let o = Offsets::from(c);
            match (o.x, o.y) {
                (HorizontalOffset::Left, VerticalOffset::Center) => '◀',
                (HorizontalOffset::Right, VerticalOffset::Center) => '▶',
                (HorizontalOffset::Center, VerticalOffset::Top) => '▲',
                (HorizontalOffset::Center, VerticalOffset::Bottom) => '▼',
                (HorizontalOffset::Left, VerticalOffset::Top) => '◤',
                (HorizontalOffset::Right, VerticalOffset::Top) => '◥',
                (HorizontalOffset::Left, VerticalOffset::Bottom) => '◣',
                (HorizontalOffset::Right, VerticalOffset::Bottom) => '◢',
                _ => panic!("Unexpected slide offset {:?}", c),
            }
        }
        Some(TileAction::Command) => '?',
        Some(TileAction::JumpSlide) => '⃤', // TODO this should depend on position
        Some(TileAction::Strike) => '☆',
        Some(TileAction::Unit) => match side {
            CurrentSide::Initial => '♟',
            CurrentSide::Flipped => '♙',
        },
        None => '-',
    }
}
