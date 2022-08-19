extern crate fstrings;

use std::{io, thread};
use std::borrow::{Borrow, BorrowMut};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::game::ai::alpha_beta::HeuristicAlphaBetaPlayer;
use crate::game::ai::stupid_sync_ai::StupidSyncAi;
use crate::game::bag::TileBag;
use crate::game::board_setup::{DukeInitialLocation, FootmenSetup};
use crate::game::state::{GameResult, GameState};
use crate::game::tile::{Owner, TileRef};
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
            TileRef::new(units::knight()),
            TileRef::new(units::champion()),
            TileRef::new(units::bowman()),
            TileRef::new(units::priest()),
            TileRef::new(units::wizard()),
            // TileRef::new(units::marshall()),
            // TileRef::new(units::general()),
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
    let smart_ai = HeuristicAlphaBetaPlayer::all_heuristics_with_max_depth(2);
    loop {
        // TODO some kind of logging mechanism
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
        let winner = controller.borrow().game_result();
        let message = match winner {
            GameResult::Tie => Some("The game ended in a tie!\nPress any key to quit".to_owned()),
            GameResult::Ongoing => None,
            GameResult::Won(w) => Some(format!("{} Won! Game is over!\nPress any key to quit", w)),
        };
        if let Some(msg) = message {
            controller.borrow_mut().add_info(msg.as_str());
            // TODO reduce duplication with above
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
        }
        match rx.recv()? {
            Event::Input(event) => match event.code {
                KeyCode::Char('q') => break,
                KeyCode::Char('h') => wrap!(controller, ControllerCommand::Left),
                KeyCode::Char('j') => wrap!(controller, ControllerCommand::Down),
                KeyCode::Char('k') => wrap!(controller, ControllerCommand::Up),
                KeyCode::Char('l') => wrap!(controller, ControllerCommand::Right),
                KeyCode::Char('p') => wrap!(controller, ControllerCommand::PullFromBag),
                KeyCode::Char('b') => wrap!(controller, ControllerCommand::CurrentOwnerBag),
                KeyCode::Char('B') => wrap!(controller, ControllerCommand::OtherOwnerBag),
                KeyCode::Char('u') => wrap!(controller, ControllerCommand::Undo),
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
