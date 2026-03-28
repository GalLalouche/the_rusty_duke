#![feature(backtrace)]
extern crate fstrings;

use std::panic;

use backtrace::Backtrace;

pub use duke_rust::common;
pub use duke_rust::game;

// Re-export macros so that `crate::assert_not!` etc. still work in the view module
pub use duke_rust::assert_not;
pub use duke_rust::assert_none;
pub use duke_rust::assert_some;

mod view;

fn main() -> () {
    // panic::set_hook(Box::new(|panic_info| {
    //     println!("{:?}", Backtrace::new());
    // }));

    crate::view::tui::main::go_main();
}
