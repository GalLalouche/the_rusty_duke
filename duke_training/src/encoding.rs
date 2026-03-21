use burn::prelude::*;
use duke_rust::game::state::GameState;
use duke_rust::game::tile::CurrentSide;

/// Number of input planes for the board encoding.
///
/// 13 tile types x 2 (current player / opponent) = 26 planes for tile identity,
/// plus 4 planes for side info:
///   - current player's tile on initial side
///   - current player's tile on flipped side
///   - opponent's tile on initial side
///   - opponent's tile on flipped side
///
/// Total: 30 planes, each 6x6.
pub const NUM_PLANES: usize = 30;
pub const BOARD_SIZE: usize = 6;

/// The 13 tile types currently supported in the game.
const TILE_NAMES: [&str; 13] = [
    "Duke",
    "Footman",
    "Pikeman",
    "Knight",
    "Champion",
    "Dragoon",
    "Wizard",
    "General",
    "Marshall",
    "Assassin",
    "Priest",
    "Bowman",
    "Longbowman",
];

pub fn tile_name_to_index(name: &str) -> Option<usize> {
    TILE_NAMES.iter().position(|&n| n == name)
}

/// Encode a `GameState` into a tensor of shape `[30, 6, 6]`.
///
/// The encoding is relative to the current player ("my" vs "opponent").
/// - Planes 0..12:  1.0 where current player's tile of type i is present
/// - Planes 13..25: 1.0 where opponent's tile of type i is present
/// - Plane 26: current player's tile on initial side
/// - Plane 27: current player's tile on flipped side
/// - Plane 28: opponent's tile on initial side
/// - Plane 29: opponent's tile on flipped side
///
/// Delegates to `encode_state_flat` and reshapes, ensuring all encoding
/// paths share the same underlying logic.
pub fn encode_state<B: Backend>(gs: &GameState, device: &B::Device) -> Tensor<B, 3> {
    encode_state_flat(gs, device)
        .reshape([NUM_PLANES as i32, BOARD_SIZE as i32, BOARD_SIZE as i32])
}

/// Returns the indices of active features in the sparse encoding.
/// Each tile contributes 2 features: one tile-type plane index and one side plane index.
/// Max ~24 active features (12 tiles x 2 features each).
pub fn active_feature_indices(gs: &GameState) -> Vec<usize> {
    let current_player = gs.current_player_turn();
    let board = gs.board();
    let mut features = Vec::with_capacity(24);

    for (coords, placed_tile) in board.active_coordinates() {
        let x = coords.x as usize;
        let y = coords.y as usize;
        let cell_index = y * BOARD_SIZE + x;
        let is_mine = placed_tile.owner == current_player;

        if let Some(tile_idx) = tile_name_to_index(placed_tile.tile.get_name()) {
            let plane = if is_mine { tile_idx } else { tile_idx + 13 };
            features.push(plane * BOARD_SIZE * BOARD_SIZE + cell_index);
        }

        let side_plane = match (is_mine, placed_tile.current_side) {
            (true, CurrentSide::Initial) => 26,
            (true, CurrentSide::Flipped) => 27,
            (false, CurrentSide::Initial) => 28,
            (false, CurrentSide::Flipped) => 29,
        };
        features.push(side_plane * BOARD_SIZE * BOARD_SIZE + cell_index);
    }

    features
}

/// Encode a `GameState` into a flat tensor of shape `[1080]` (= 30 * 6 * 6).
///
/// This builds on `active_feature_indices` to ensure encoding consistency
/// between the burn training path and the NNUE inference path.
pub fn encode_state_flat<B: Backend>(gs: &GameState, device: &B::Device) -> Tensor<B, 1> {
    let total = NUM_PLANES * BOARD_SIZE * BOARD_SIZE;
    let mut data = vec![0.0f32; total];

    for idx in active_feature_indices(gs) {
        debug_assert!(idx < total, "Feature index {} out of bounds", idx);
        data[idx] = 1.0;
    }

    Tensor::<B, 1>::from_floats(data.as_slice(), device)
}
