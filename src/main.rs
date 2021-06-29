#[macro_use(f)]
extern crate fstrings;

use std::ops::Deref;
use std::sync::Arc;

use pancurses::{endwin, initscr};


// use nwg::NativeUi;
use crate::game::state::{DukeInitialLocation, FootmenSetup, GameState};
use crate::game::token::TokenBag;

mod common;
mod game;
mod view;
// mod units;

// #![windows_subsystem = "windows"]
// /*!
//     A very simple application that shows your name in a message box.
//     Uses layouts to position the controls in the window
//  */

// extern crate native_windows_gui as nwg;

// #[derive(Default)]
// pub struct BasicApp {
//     window: nwg::Window,
//     layout: nwg::GridLayout,
//     name_edit: nwg::TextInput,
//     hello_button: nwg::Button,
// }
//
// impl BasicApp {
//     fn say_hello(&self) {
//         nwg::modal_info_message(&self.window, "Hello", &format!("Hello {}", self.name_edit.text()));
//     }
//
//     fn say_goodbye(&self) {
//         nwg::modal_info_message(&self.window, "Goodbye", &format!("Goodbye {}", self.name_edit.text()));
//         nwg::stop_thread_dispatch();
//     }
// }
//
// //
// // ALL of this stuff is handled by native-windows-derive
// //
// // mod basic_app_ui {
// //     use native_windows_gui as nwg;
// //     use super::*;
// //     use std::rc::Rc;
// //     use std::cell::RefCell;
// //     use std::ops::Deref;
// //
// //     pub struct BasicAppUi {
// //         inner: Rc<BasicApp>,
// //         default_handler: RefCell<Option<nwg::EventHandler>>
// //     }
// //
// //     impl nwg::NativeUi<BasicAppUi> for BasicApp {
// //         fn build_ui(mut data: BasicApp) -> Result<BasicAppUi, nwg::NwgError> {
// //             use nwg::Event as E;
// //
// //             // Controls
// //             nwg::Window::builder()
// //                 .flags(nwg::WindowFlags::WINDOW | nwg::WindowFlags::VISIBLE)
// //                 .size((300, 115))
// //                 .position((300, 300))
// //                 .title("Basic example")
// //                 .build(&mut data.window)?;
// //
// //             nwg::TextInput::builder()
// //                 .text("Heisenberg")
// //                 .parent(&data.window)
// //                 .focus(true)
// //                 .build(&mut data.name_edit)?;
// //
// //             nwg::Button::builder()
// //                 .text("Say my name")
// //                 .parent(&data.window)
// //                 .build(&mut data.hello_button)?;
// //
// //             // Wrap-up
// //             let ui = BasicAppUi {
// //                 inner: Rc::new(data),
// //                 default_handler: Default::default(),
// //             };
// //
// //             // Events
// //             let evt_ui = Rc::downgrade(&ui.inner);
// //             let handle_events = move |evt, _evt_data, handle| {
// //                 if let Some(ui) = evt_ui.upgrade() {
// //                     match evt {
// //                         E::OnButtonClick =>
// //                             if &handle == &ui.hello_button {
// //                                 BasicApp::say_hello(&ui);
// //                             },
// //                         E::OnWindowClose =>
// //                             if &handle == &ui.window {
// //                                 BasicApp::say_goodbye(&ui);
// //                             },
// //                         _ => {}
// //                     }
// //                 }
// //             };
// //
// //             *ui.default_handler.borrow_mut() = Some(nwg::full_bind_event_handler(&ui.window.handle, handle_events));
// //
// //             // Layouts
// //             nwg::GridLayout::builder()
// //                 .parent(&ui.window)
// //                 .spacing(1)
// //                 .child(0, 0, &ui.name_edit)
// //                 .child_item(nwg::GridLayoutItem::new(&ui.hello_button, 0, 1, 1, 2))
// //                 .build(&ui.layout)?;
// //
// //             return Ok(ui);
// //         }
// //     }
// //
// //     impl Drop for BasicAppUi {
// //         /// To make sure that everything is freed without issues, the default handler must be unbound.
// //         fn drop(&mut self) {
// //             let handler = self.default_handler.borrow();
// //             if handler.is_some() {
// //                 nwg::unbind_event_handler(handler.as_ref().unwrap());
// //             }
// //         }
// //     }
// //
// //     impl Deref for BasicAppUi {
// //         type Target = BasicApp;
// //
// //         fn deref(&self) -> &BasicApp {
// //             &self.inner
// //         }
// //     }
// // }
//
// struct DoryanState {
//     state: GameState,
// }
//
// impl Engine for DoryanState {
//     fn render(&mut self, api: &mut dyn DoryenApi) {
//         let con = api.con();
//         con.rectangle(
//             0,
//             0,
//             100,
//             100,
//             Some((128, 128, 128, 255)),
//             Some((0, 0, 0, 255)),
//             Some('*' as u16),
//         );
//         con.area(
//             10,
//             10,
//             5,
//             5,
//             Some((255, 64, 64, 255)),
//             Some((128, 32, 32, 255)),
//             Some('%' as u16),
//         );
//
//         con.print_color(
//             5,
//             5,
//             "#[blue]This blue text contains a #[red]red#[] word",
//             TextAlign::Left,
//             None,
//         );
//     }
// }
//
// fn main() {
//     let mut app = App::new(AppOptions {
//         console_width: 100,
//         console_height: 100,
//         screen_width: 100 * 8,
//         screen_height: 100 * 8,
//         window_title: "Zi Rusty Duke".to_owned(),
//         font_path: "terminal_8x8.png".to_owned(),
//         vsync: true,
//         fullscreen: false,
//         show_cursor: true,
//         resizable: true,
//         intercept_close_request: false,
//     });
//     app.set_engine(Box::new(DoryanState {
//         state: GameState::new(
//             &TokenBag::new(Vec::new()),
//             (DukeInitialLocation::Left,
//              FootmenSetup::Sides),
//             (DukeInitialLocation::Left,
//              FootmenSetup::Sides),
//         )
//     }));
//     app.run();
//     // nwg::init().expect("Failed to init Native Windows GUI");
//     // nwg::Font::set_global_family("Segoe UI").expect("Failed to set default font");
//     // let _ui = BasicApp::build_ui(Default::default()).expect("Failed to build UI");
//     // nwg::dispatch_thread_events();
//     // use rltk::RltkBuilder;
//     // let context: _ = RltkBuilder::simple80x50()
//     //     .with_title("Roguelike Tutorial")
//     //     .build()?;
// }

fn main() {
    let window = initscr();
    window.printw("Hello Rust");
    window.refresh();
    window.getch();
    endwin();
}
