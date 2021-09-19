extern crate fstrings;

use std::{io, thread};
use std::borrow::{Borrow, BorrowMut};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::game::ai::heuristic_ai::HeuristicAi;
use crate::game::ai::heuristics;
use crate::game::ai::stupid_sync_ai::StupidSyncAi;
use crate::game::board_setup::{DukeInitialLocation, FootmenSetup};
use crate::game::state::GameState;
use crate::game::tile::{Owner, TileBag, TileRef};
use crate::game::units;
use crate::view::controller::{Controller, ControllerCommand};
use crate::view::state::ViewState;

enum Event<I> {
    Input(I),
    Tick,
}

pub fn go_main() -> Result<(), Box<dyn std::error::Error>> {
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
        &TileBag::new(vec!(
            TileRef::new(units::footman()),
            TileRef::new(units::footman()),
            TileRef::new(units::pikeman()),
            TileRef::new(units::pikeman()),
        )),
        (DukeInitialLocation::Left, FootmenSetup::Left),
        (DukeInitialLocation::Right, FootmenSetup::Right),
    );
    let vs = ViewState::new(gs);
    let mut controller = Controller::new(vs);
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
    let dumb_ai = StupidSyncAi {};
    let smart_ai = HeuristicAi::new(
        Owner::BottomPlayer,
        1,
        todo!(),
        // vec!(
        //     Heuristics::duke_movement_options(),
        //     heuristics::total_tiles_on_board(),
        // ),
    );
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

            rect.render_widget(controller.borrow(), chunks[0]);
        })?;
        macro_rules! wrap {
            ($controller: ident, $command: expr) => {
                match $controller.apply($command) {
                    None => (),
                    Some(e) => $controller.add_info(e.error_msg().as_str()),
                }
            }
        }
        match rx.recv()? {
            Event::Input(event) => match event.code {
                KeyCode::Char('q') => break,
                KeyCode::Char('h') => wrap!(controller, ControllerCommand::Left),
                KeyCode::Char('j') => wrap!(controller, ControllerCommand::Down),
                KeyCode::Char('k') => wrap!(controller, ControllerCommand::Up),
                KeyCode::Char('l') => wrap!(controller, ControllerCommand::Right),
                KeyCode::Char('p') => wrap!(controller, ControllerCommand::PullFromBag),
                KeyCode::Char('u') => {
                    unimplemented!("Undo is not supported");
                }
                KeyCode::Enter => wrap!(controller, ControllerCommand::Select),
                KeyCode::Esc => wrap!(controller, ControllerCommand::Escape),
                KeyCode::Char('n') => {
                    match controller.current_player_turn() {
                        Owner::TopPlayer => controller.ai_move(&dumb_ai),
                        Owner::BottomPlayer => controller.ai_move(&smart_ai),
                    }
                }
                _ => {}
            },
            Event::Tick => {}
        }
    };
    Ok(())
}
