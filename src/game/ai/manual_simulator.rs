use std::io;
use std::io::Read;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::game::ai::ai_move::ArtificialPlayer;
use crate::game::ai::heuristic_ai::HeuristicAi;
use crate::game::ai::heuristics;
use crate::game::ai::stupid_sync_ai::StupidSyncAi;
use crate::game::board::{DukeInitialLocation, FootmenSetup};
use crate::game::state::GameState;
use crate::game::tile::{Owner, TileBag, TileRef};
use crate::game::units;
use crate::game::ai::heuristics::Heuristics;

fn go_aux(turn_count: i32, max_depth: usize, print: bool, wait_for_input: bool) {
    let mut gs = GameState::new(
        &TileBag::new(vec!(
            TileRef::new(units::footman()),
            TileRef::new(units::footman()),
            TileRef::new(units::pikeman()),
            TileRef::new(units::pikeman()),
        )),
        (DukeInitialLocation::Left, FootmenSetup::Left),
        (DukeInitialLocation::Right, FootmenSetup::Right),
    );

    let dumb_ai = StupidSyncAi {};
    let smart_ai = HeuristicAi::new(
        Owner::BottomPlayer,
        max_depth,
        vec!(
            Box::new(Heuristics::DukeMovementOptions),
            Box::new(Heuristics::TotalTilesOnBoard),
            Box::new(Heuristics::TotalMovementOptions),
        ),
    );

    let mut std_gen = StdRng::seed_from_u64(0);
    for _ in 0..(if turn_count > 0 { turn_count } else { 10000000 }) {
        if print {
            println!("{}", gs.as_string());
        }
        if wait_for_input {
            io::stdin().bytes().next();
        }
        match gs.current_player_turn {
            Owner::TopPlayer => dumb_ai.play_next_move(&mut std_gen, &mut gs),
            Owner::BottomPlayer => smart_ai.play_next_move(&mut std_gen, &mut gs)
        }
    }
}

pub fn go_main() {
    go_aux(-1, 2, true, true);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_does_not_throw() {
        go_aux(2, 1, false, false);
    }
}