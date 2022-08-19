#![feature(backtrace)]
extern crate fstrings;

use std::panic;

use backtrace::Backtrace;


mod common;
mod game;
mod view;

fn main() -> () {
    panic::set_hook(Box::new(|panic_info| {
        println!("{:?}", Backtrace::new());
    }));

    crate::view::tui::main::go_main();
}
