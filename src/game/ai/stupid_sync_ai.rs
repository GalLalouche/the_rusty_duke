use std::borrow::BorrowMut;

use rand::Rng;
use rand::seq::SliceRandom;

use crate::game::ai::ai_move::ArtificialPlayer;
use crate::game::state::GameState;

pub struct StupidSyncAi {}

impl ArtificialPlayer for StupidSyncAi {
    fn play_next_move<R>(&self, rng: &mut R, gs: &mut GameState) -> () where R: Rng {
        let pms = gs.all_valid_game_moves_for_current_player();
        // From https://stackoverflow.com/a/34215930/736508
        let pm = pms.choose(rng.borrow_mut());
        gs.make_a_move(pm.expect("Cannot find a simple dumb move... is the game over? :(").into());
    }
}