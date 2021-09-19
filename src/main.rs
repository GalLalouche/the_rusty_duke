extern crate fstrings;


mod common;
mod game;
mod view;

fn main() -> () {
    crate::game::ai::manual_simulator::go_main()
}
