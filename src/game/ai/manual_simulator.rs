use std::io;
use std::io::Read;

use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::game::ai::alpha_beta::HeuristicAlphaBetaPlayer;

use crate::game::ai::player::ArtificialPlayer;
use crate::game::ai::stupid_sync_ai::StupidSyncAi;
use crate::game::bag::TileBag;
use crate::game::board_setup::{DukeInitialLocation, FootmenSetup};
use crate::game::state::GameState;
use crate::game::tile::{Owner, TileRef};
use crate::game::units;

fn go_aux(turn_count: u32, max_depth: u32, print: bool, wait_for_input: bool) {
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
    let smart_ai = HeuristicAlphaBetaPlayer::all_heuristics_with_max_depth(max_depth);

    let mut std_gen = StdRng::seed_from_u64(0);
    for _ in 0..(if turn_count > 0 { turn_count } else { 10000000 }) {
        if print {
            println!("{}", gs.as_double_string());
        }
        if wait_for_input {
            io::stdin().bytes().next();
        }
        match gs.current_player_turn() {
            Owner::TopPlayer => dumb_ai.play_next_move(&mut std_gen, &mut gs),
            Owner::BottomPlayer => smart_ai.play_next_move(&mut std_gen, &mut gs)
        };
    }
}

pub fn go_main() {
    go_aux(3, 3, true, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_does_not_throw() {
        go_aux(2, 1, false, false);
    }
}