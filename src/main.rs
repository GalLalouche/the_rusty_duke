extern crate fstrings;

use std::{io, thread};
use std::borrow::Borrow;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::view::tui::move_view::MoveView;

use crate::game::board::{DukeInitialLocation, FootmenSetup};
use crate::game::state::GameState;
use crate::game::tile::TileBag;
use crate::game::units;
use crate::view::state::ViewState;

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

    let gs = GameState::new(
        &TileBag::new(vec!(units::footman(), units::footman(), units::pikeman(), units::pikeman())),
        (DukeInitialLocation::Left, FootmenSetup::Right),
        (DukeInitialLocation::Right, FootmenSetup::Right),
    );
    let mut vs = ViewState::new(gs);
    // stupid_sync_ai::next_move(gs.borrow_mut());

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

    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    loop {
        // TODO some kind of logging mechanism
        // TODO some kind of info/warning mechanism (e.g., cannot unplace tile)
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

            rect.render_widget(vs.borrow(), chunks[0]);
        })?;
        macro_rules! place {
            ($e: expr, $terminal: ident, $vs: ident) => {
                if $vs.can_move_placement($e) {
                    $vs.move_placement($e);
                } else if $vs.can_move_view_position($e) {
                    $vs.move_view_position($e);
                }
            }
        }
        match rx.recv()? {
            Event::Input(event) => match event.code {
                KeyCode::Char('h') => place!(MoveView::Left, terminal, vs),
                KeyCode::Char('j') => place!(MoveView::Down, terminal, vs),
                KeyCode::Char('k') => place!(MoveView::Up, terminal, vs),
                KeyCode::Char('l') => place!(MoveView::Right, terminal, vs),
                KeyCode::Char('p') => {
                    // disable_raw_mode()?;
                    // terminal.show_cursor()?;
                    vs.pull_token_from_bag();
                }
                KeyCode::Char('u') => {
                    unimplemented!("Undo is not supported");
                }
                KeyCode::Enter => {
                    if vs.is_moving() {
                        vs.move_selected();
                    } else if vs.is_placing() {
                        vs.place();
                    } else if vs.can_select_for_movement() {
                        vs.select_for_movement();
                    }
                }
                KeyCode::Esc => {
                    if vs.can_unselect() {
                        vs.unselect();
                    }
                }
                KeyCode::Char('q') => {
                    break;
                }
                _ => {}
            },
            Event::Tick => {}
        }
    };
    Ok(())
}
