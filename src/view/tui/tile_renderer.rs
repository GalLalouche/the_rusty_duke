use crossterm::style::Stylize;
use tui::buffer::Buffer;
use tui::layout::Rect;
use tui::style::{Color, Modifier, Style};
use tui::widgets::{Block, Borders, BorderType, Widget};

use crate::common::coordinates::Coordinates;
use crate::game::offset::{HorizontalOffset, Offsets, VerticalOffset};
use crate::game::tile::{CurrentSide, Owner, PlacedTile};
use crate::game::tile::Owner::TopPlayer;
use crate::game::tile_side::TileAction;

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
                // TODO handle diagonal slides
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
    let b = Block::default()
        .borders(Borders::ALL)
        .border_type(if highlight { BorderType::Thick } else { BorderType::Plain })
        .border_style(Style::default().fg(color))
        ;
    let with_title_maybe = if let Some(o) = o {
        let color = match o.owner {
            TopPlayer => Color::LightCyan,
            Owner::BottomPlayer => Color::LightMagenta,
        };
        b.title(match config {
            RenderTileConfig::Info { title } => title,
            _ => o.tile.get_name().clone(),
        }).style(Style::default().bg(color))
    } else {
        b
    };
    let inside_border = with_title_maybe.inner(area);
    with_title_maybe.render(area, buf);
    if let Some(o) = o {
        for (c, t) in o.get_current_side().get_board().all_coordinated_values() {
            let normalized_c: Coordinates = c.into();
            buf.set_string(
                inside_border.x + normalized_c.x as u16,
                inside_border.y + normalized_c.y as u16,
                to_char(c, t.cloned(), o.current_side).to_string(), Style::default(),
            );
        }
    }
}
