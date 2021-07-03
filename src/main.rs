extern crate fstrings;

use std::{io, thread};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use tui::buffer::Buffer;
use tui::layout::Rect;
use tui::style::Style;
use tui::widgets::{Block, Borders, BorderType, Widget};

use crate::common::coordinates::Coordinates;
use crate::game::board::GameBoard;
use crate::game::offset::{HorizontalOffset, Offsets, VerticalOffset};
use crate::game::state::{DukeInitialLocation, FootmenSetup, GameState};
use crate::game::token::{CurrentSide, OwnedToken, TokenAction, TokenBag};
use crate::game::token::Owner::Player1;
use crate::game::units;

mod common;
mod game;
mod view;


enum Event<I> {
    Input(I),
    Tick,
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    use tui::Terminal;
    use tui::backend::CrosstermBackend;
    use tui::layout::{Layout, Constraint, Direction};

    let (tx, rx) = mpsc::channel();
    let tick_rate = Duration::from_millis(200);
    use crossterm::{
        event::{self, Event as CEvent, KeyCode},
        terminal::{disable_raw_mode, enable_raw_mode},
    };

    enable_raw_mode().expect("can run in raw mode");

    thread::spawn(move || {
        let mut last_tick = Instant::now();
        loop {
            let timeout = tick_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));

            if event::poll(timeout).expect("poll works") {
                if let CEvent::Key(key) = event::read().expect("can read events") {
                    tx.send(Event::Input(key)).expect("can send events");
                }
            }

            if last_tick.elapsed() >= tick_rate {
                if let Ok(_) = tx.send(Event::Tick) {
                    last_tick = Instant::now();
                }
            }
        }
    });

    println!("{:?}", units::duke(Player1).token.get_current_side().actions());
    println!("{:?}", units::footman(Player1).token.get_current_side().actions());
    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    loop {
        terminal.draw(|rect| {
            let size = rect.size();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(2)
                .constraints(
                    [
                        Constraint::Length(100),
                        Constraint::Length(3),
                        Constraint::Min(2),
                        Constraint::Length(3),
                    ]
                        .as_ref(),
                )
                .split(size);

            let gs = GameState::new(
                &TokenBag::new(Vec::new()),
                (DukeInitialLocation::Left, FootmenSetup::Right),
                (DukeInitialLocation::Right, FootmenSetup::Right),
            );

            rect.render_widget(gs.board, chunks[0]);
        });

        match rx.recv()? {
            Event::Input(event) => match event.code {
                KeyCode::Char('q') => {
                    disable_raw_mode()?;
                    terminal.show_cursor()?;
                    break;
                }
                _ => {}
            },
            Event::Tick => {}
        }
    };
    Ok(())
}
