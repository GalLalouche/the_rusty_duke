use std::borrow::BorrowMut;

use rand::Rng;
use rand::seq::SliceRandom;

use crate::game::ai::player::{AiMove, ArtificialPlayer};
use crate::game::state::GameState;

pub struct StupidSyncAi {}

impl ArtificialPlayer for StupidSyncAi {
    fn get_next_move<R>(&self, rng: &mut R, gs: &GameState) -> AiMove where R: Rng {
        let pms = gs.all_valid_game_moves_for_current_player().collect::<Vec<_>>();
        // From https://stackoverflow.com/a/34215930/736508
        pms.choose(rng.borrow_mut()).unwrap().into()
    }

    fn create(_max_depth: u32) -> Self {
        StupidSyncAi {}
    }
}