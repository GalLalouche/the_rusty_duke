//! 64-bit fingerprints for deduplication and (future) transposition tables.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use duke_rust::game::state::GameState;
use duke_rust::game::tile::Owner;

/// Hash of board layout, side to move, bags, and discards (no heap allocation).
///
/// Compatible with the previous `Vec` + `sort` implementation: same multiset of
/// tile-type bytes is sorted and hashed identically.
pub fn position_key(gs: &GameState) -> u64 {
    let mut h = DefaultHasher::new();

    for y in 0..6u8 {
        for x in 0..6u8 {
            let c = duke_rust::common::coordinates::Coordinates { x, y };
            match gs.board().get(c) {
                Some(t) => {
                    1u8.hash(&mut h);
                    t.tile_type.hash(&mut h);
                    t.owner.hash(&mut h);
                    t.current_side.hash(&mut h);
                }
                None => 0u8.hash(&mut h),
            }
        }
    }

    gs.current_player_turn().hash(&mut h);

    let mut scratch = [0u8; 16];
    for (owner, max_n) in [
        (Owner::TopPlayer, 12usize),
        (Owner::BottomPlayer, 12usize),
    ] {
        let r = gs.bag_for_owner(owner).remaining();
        let n = r.len().min(max_n);
        for (i, t) in r.iter().take(n).enumerate() {
            scratch[i] = *t as u8;
        }
        scratch[..n].sort_unstable();
        scratch[..n].hash(&mut h);
    }
    for (owner, max_n) in [
        (Owner::TopPlayer, 16usize),
        (Owner::BottomPlayer, 16usize),
    ] {
        let r = gs.discard_bag_for(owner).existing();
        let n = r.len().min(max_n);
        for (i, t) in r.iter().take(n).enumerate() {
            scratch[i] = *t as u8;
        }
        scratch[..n].sort_unstable();
        scratch[..n].hash(&mut h);
    }

    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_setup::{create_bag, create_initial_state};

    #[test]
    fn position_key_nonzero_on_start_position() {
        let bag = create_bag();
        let gs = create_initial_state(&bag);
        assert_ne!(position_key(&gs), 0);
    }
}
