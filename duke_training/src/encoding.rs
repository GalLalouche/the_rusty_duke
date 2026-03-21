use burn::prelude::*;
use duke_rust::game::state::GameState;
use duke_rust::game::tile::{CurrentSide, TileType};
use strum::EnumCount;

pub const NUM_TILE_TYPES: usize = TileType::COUNT;
/// 13 tile types × 2 (my/opponent) + 4 side planes = 30
pub const NUM_PLANES: usize = NUM_TILE_TYPES * 2 + 4;
pub const BOARD_SIZE: usize = 6;

/// Encode a `GameState` into a tensor of shape `[NUM_PLANES, 6, 6]`.
///
/// Encoding is relative to the current player:
/// - Planes 0..12:  current player's tile of type i
/// - Planes 13..25: opponent's tile of type i
/// - Plane 26/27: current player's tile on initial/flipped side
/// - Plane 28/29: opponent's tile on initial/flipped side
pub fn encode_state<B: Backend>(gs: &GameState, device: &B::Device) -> Tensor<B, 3> {
    encode_state_flat(gs, device)
        .reshape([NUM_PLANES as i32, BOARD_SIZE as i32, BOARD_SIZE as i32])
}

/// Returns indices of active features in the sparse encoding.
/// Each tile contributes 2 features (type plane + side plane).
pub fn active_feature_indices(gs: &GameState) -> Vec<usize> {
    let current_player = gs.current_player_turn();
    let board = gs.board();
    let mut features = Vec::with_capacity(24);

    for (coords, placed_tile) in board.active_coordinates() {
        let cell = coords.y as usize * BOARD_SIZE + coords.x as usize;
        let is_mine = placed_tile.owner == current_player;
        let tile_idx = placed_tile.tile.tile_type().index();

        let type_plane = if is_mine { tile_idx } else { tile_idx + NUM_TILE_TYPES };
        features.push(type_plane * BOARD_SIZE * BOARD_SIZE + cell);

        let side_plane = match (is_mine, placed_tile.current_side) {
            (true, CurrentSide::Initial) => NUM_TILE_TYPES * 2,
            (true, CurrentSide::Flipped) => NUM_TILE_TYPES * 2 + 1,
            (false, CurrentSide::Initial) => NUM_TILE_TYPES * 2 + 2,
            (false, CurrentSide::Flipped) => NUM_TILE_TYPES * 2 + 3,
        };
        features.push(side_plane * BOARD_SIZE * BOARD_SIZE + cell);
    }

    features
}

/// Flat tensor of shape `[1080]`. Delegates to `active_feature_indices`.
pub fn encode_state_flat<B: Backend>(gs: &GameState, device: &B::Device) -> Tensor<B, 1> {
    let total = NUM_PLANES * BOARD_SIZE * BOARD_SIZE;
    let mut data = vec![0.0f32; total];

    for idx in active_feature_indices(gs) {
        debug_assert!(idx < total);
        data[idx] = 1.0;
    }

    Tensor::<B, 1>::from_floats(data.as_slice(), device)
}
