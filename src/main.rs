extern crate fstrings;

use std::borrow::Borrow;

use common::timer::GLOBAL_TIMERS;
use std::collections::HashMap;

mod common;
mod game;
mod view;

fn main() -> () {
    crate::game::ai::manual_simulator::go_main();

    GLOBAL_TIMERS.with(|map| {
        let mut v: HashMap<&str, i64> = map.take().clone();
        let mut v2: Vec<_> = v.into_iter().collect();
        let total: f64 = v2.iter().map(|e| e.1 as f64).sum();
        v2.sort_by_key(|e| -e.1);
        for (k, s) in v2 {
            println!(
                "Total time for {} was {} ms ({:.2} %)",
                k,
                (s as f64) / 1000000.,
                ((s as f64) / total) * 100.0,
            );
        }
    })
}
