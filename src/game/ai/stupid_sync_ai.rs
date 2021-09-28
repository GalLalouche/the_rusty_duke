use std::borrow::BorrowMut;

use rand::Rng;
use rand::seq::SliceRandom;

use crate::game::ai::player::{AiMove, ArtificialPlayer};
use crate::game::state::GameState;
use crate::game::tile::Owner;

pub struct StupidSyncAi {}

impl ArtificialPlayer for StupidSyncAi {
    fn get_next_move<R>(&self, rng: &mut R, gs: &GameState) -> AiMove where R: Rng {
        let pms = gs.all_valid_game_moves_for_current_player();
        // From https://stackoverflow.com/a/34215930/736508
        pms.choose(rng.borrow_mut()).unwrap().into()
    }

    fn create(max_depth: u32) -> Self {
        StupidSyncAi {}
    }
}