use burn::prelude::*;
use duke_rust::game::state::GameState;
use duke_rust::game::tile::{CurrentSide, TileType};
use strum::EnumCount;

pub const NUM_TILE_TYPES: usize = TileType::COUNT;
pub const BOARD_SIZE: usize = 6;
/// Board planes: 13 tile types × 2 (my/opponent) + 4 side planes = 30
pub const NUM_BOARD_PLANES: usize = NUM_TILE_TYPES * 2 + 4;
pub const BOARD_FEATURES: usize = NUM_BOARD_PLANES * BOARD_SIZE * BOARD_SIZE; // 1080
/// Bag features: 13 tile types × 2 (my bag / opponent bag) = 26 scalars
pub const BAG_FEATURES: usize = NUM_TILE_TYPES * 2;
/// Max board features: up to BOARD_SIZE^2 tiles on board, each producing 2 features (type + side).
pub const MAX_BOARD_FEATURE_COUNT: usize = BOARD_SIZE * BOARD_SIZE * 2;
/// Total input size for the flat FC model.
pub const TOTAL_FEATURES: usize = BOARD_FEATURES + BAG_FEATURES; // 1106

/// Stack-allocated feature buffer. Avoids heap allocation in hot path.
pub struct FeatureBuffer {
    pub data: [usize; MAX_BOARD_FEATURE_COUNT],
    pub len: usize,
}

impl FeatureBuffer {
    #[inline]
    fn new() -> Self {
        Self { data: [0; MAX_BOARD_FEATURE_COUNT], len: 0 }
    }

    #[inline]
    fn push(&mut self, val: usize) {
        debug_assert!(self.len < MAX_BOARD_FEATURE_COUNT,
            "FeatureBuffer overflow: tried to push {} features (max {})",
            self.len + 1, MAX_BOARD_FEATURE_COUNT);
        self.data[self.len] = val;
        self.len += 1;
    }

    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.data[..self.len]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a FeatureBuffer {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

/// Board feature indices (sparse binary). Returns stack-allocated buffer.
pub fn active_board_features(gs: &GameState) -> FeatureBuffer {
    let current_player = gs.current_player_turn();
    let board = gs.board();
    let mut features = FeatureBuffer::new();

    for (coords, placed_tile) in board.active_coordinates() {
        let cell = coords.y as usize * BOARD_SIZE + coords.x as usize;
        debug_assert!(cell < BOARD_SIZE * BOARD_SIZE,
            "Board coordinate ({}, {}) maps to cell {} which is outside the {}x{} board",
            coords.x, coords.y, cell, BOARD_SIZE, BOARD_SIZE);
        let is_mine = placed_tile.owner == current_player;
        let tile_idx = placed_tile.tile_type.index();

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

/// Bag tile counts: [my_bag_type_0, ..., my_bag_type_12, opp_bag_type_0, ..., opp_bag_type_12]
/// Each value is the count of that tile type in the respective bag.
pub fn bag_features(gs: &GameState) -> [f32; BAG_FEATURES] {
    let current_player = gs.current_player_turn();
    let my_bag = gs.bag_for_current_player();
    let opp_bag = gs.bag_for_other_player();

    let mut features = [0.0f32; BAG_FEATURES];

    for tile in my_bag.remaining() {
        let idx = tile.index();
        debug_assert!(idx < NUM_TILE_TYPES, "tile type index {} >= NUM_TILE_TYPES {}", idx, NUM_TILE_TYPES);
        features[idx] += 1.0;
    }
    for tile in opp_bag.remaining() {
        let idx = tile.index();
        debug_assert!(idx < NUM_TILE_TYPES, "tile type index {} >= NUM_TILE_TYPES {}", idx, NUM_TILE_TYPES);
        features[NUM_TILE_TYPES + idx] += 1.0;
    }

    features
}

/// Backward-compatible alias returning a slice reference via FeatureBuffer.
#[inline]
pub fn active_feature_indices(gs: &GameState) -> FeatureBuffer {
    active_board_features(gs)
}

/// Encode a game state into a pre-allocated slice (avoids per-state heap allocation).
/// The slice must be exactly `TOTAL_FEATURES` long and will be zeroed and filled.
#[inline]
pub fn encode_state_flat_into(gs: &GameState, out: &mut [f32]) {
    debug_assert_eq!(out.len(), TOTAL_FEATURES);
    out.fill(0.0);

    let features = active_board_features(gs);
    for &idx in features.as_slice() {
        debug_assert!(idx < BOARD_FEATURES);
        out[idx] = 1.0;
    }

    let bag = bag_features(gs);
    out[BOARD_FEATURES..TOTAL_FEATURES].copy_from_slice(&bag);
}

/// Flat tensor of shape `[TOTAL_FEATURES]` (1106).
/// Board features (sparse binary) + bag features (dense counts).
pub fn encode_state_flat<B: Backend>(gs: &GameState, device: &B::Device) -> Tensor<B, 1> {
    let mut data = [0.0f32; TOTAL_FEATURES];
    encode_state_flat_into(gs, &mut data);
    Tensor::<B, 1>::from_floats(data.as_slice(), device)
}
