use tui::buffer::Buffer;
use tui::layout::Rect;
use tui::style::Style;
use tui::widgets::{Block, Borders, BorderType, Widget};

use crate::common::coordinates::Coordinates;
use crate::game::board::GameBoard;
use crate::game::offset::{HorizontalOffset, Offsets, VerticalOffset};
use crate::game::token::{CurrentSide, OwnedToken, TokenAction, TokenSide};

fn to_char(c: Coordinates, t: Option<&TokenAction>) -> char {
    match t {
        Some(TokenAction::Move) => '●',
        Some(TokenAction::Jump) => '○',
        Some(TokenAction::Slide) => {
            let o = Offsets::from(c);
            match (o.x, o.y) {
                (HorizontalOffset::Left, VerticalOffset::Center) => '<',
                (HorizontalOffset::Right, VerticalOffset::Center) => '>',
                (HorizontalOffset::Center, VerticalOffset::Top) => '^',
                (HorizontalOffset::Center, VerticalOffset::Bottom) => '▾',
                _ => panic!("Unexpected slide offset {:?}", c),
            }
        }
        Some(TokenAction::Command) => '?',
        Some(TokenAction::JumpSlide) => '⃤',
        Some(TokenAction::Strike) => '☆',
        None => '-'
    }
}

// Can't be a proper impl because there's no way to impl that for an Option :\
fn render_token(o: Option<&OwnedToken>, area: Rect, buf: &mut Buffer) -> () {
    let b = Block::default()
        .borders(Borders::ALL)
        ;
    let with_title_maybe = if let Some(ow) = o {
        b.title(ow.token.name.to_owned())
    } else {
        b
    };
    let inside_border = with_title_maybe.inner(area);
    with_title_maybe.render(area, buf);
    if let Some(ow) = o {
        for (c, t) in ow.token.get_current_side().get_board().all_coordinated_values() {
            let normalized_c: Coordinates = c.into();
            buf.set_string(
                inside_border.x + normalized_c.x as u16,
                inside_border.y + normalized_c.y as u16,
                to_char(c, t).to_string(), Style::default(),
            );
        }
        let center = Offsets::center_coordinates();
        buf.set_string(
            inside_border.x + center.x,
            inside_border.y + center.y,
            match ow.token.current_side {
                CurrentSide::Initial => "♙",
                CurrentSide::Flipped => "♟︎",
            },
            Style::default(),
        );
    }
}

impl Widget for GameBoard {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let b = Block::default()
            .title("Game Board")
            .border_type(BorderType::Double)
            .borders(Borders::ALL)
            ;
        let width = TokenSide::SIDE + 2;
        let height = TokenSide::SIDE + 2;
        let game_board_area = Rect {
            x: area.x,
            y: area.y,
            width: self.width() as u16 * width + 2,
            height: self.height() as u16 * height + 2,
        };
        let inner_area = b.inner(game_board_area);
        b.render(game_board_area, buf);
        for (c, w) in self.get_board().all_coordinated_values() {
            let cell_area = Rect {
                x: inner_area.x + width * c.x as u16,
                y: inner_area.y + height * c.y as u16,
                width,
                height,
            };
            render_token(w, cell_area, buf);
        }
    }
}
