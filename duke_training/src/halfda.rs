//! HalfDA (Half-Duke-Accumulator) feature encoding for NNUE-style networks.
//!
//! Each non-duke piece on the board is encoded as a sparse feature indexed by:
//!   duke_square * (36 * 13 * 2 * 2) + piece_square * (13 * 2 * 2) + piece_type * (2 * 2) + piece_color * 2 + piece_side
//!
//! Where:
//! - `duke_square`: 0-35 (current player's duke position, y*6+x)
//! - `piece_square`: 0-35 (the other piece's position)
//! - `piece_type`: 0-12 (TileType enum, 13 types)
//! - `piece_color`: 0=my piece, 1=opponent piece
//! - `piece_side`: 0=Initial, 1=Flipped
//!
//! Total features: 36 * 36 * 13 * 2 * 2 = 67,392
//! Active features per position: ~4-10 (one per non-duke piece on board)

use duke_rust::game::state::GameState;
use duke_rust::game::tile::{CurrentSide, TileType};

use crate::encoding::{BOARD_SIZE, NUM_TILE_TYPES};

/// Number of board squares (BOARD_SIZE * BOARD_SIZE).
const NUM_SQUARES: usize = BOARD_SIZE * BOARD_SIZE; // 36

/// Number of color values (my piece vs opponent piece).
const NUM_COLORS: usize = 2;

/// Number of side values (Initial vs Flipped).
const NUM_SIDES: usize = 2;

/// Total number of HalfDA features.
pub const HALFDA_FEATURES: usize = NUM_SQUARES * NUM_SQUARES * NUM_TILE_TYPES * NUM_COLORS * NUM_SIDES; // 67,392

/// Stride constants for the feature index computation.
const PIECE_SIDE_STRIDE: usize = 1;
const PIECE_COLOR_STRIDE: usize = NUM_SIDES * PIECE_SIDE_STRIDE;
const PIECE_TYPE_STRIDE: usize = NUM_COLORS * PIECE_COLOR_STRIDE;
const PIECE_SQUARE_STRIDE: usize = NUM_TILE_TYPES * PIECE_TYPE_STRIDE;
const DUKE_SQUARE_STRIDE: usize = NUM_SQUARES * PIECE_SQUARE_STRIDE;

/// Maximum number of non-duke pieces on the board.
/// In The Duke, the board is 6x6=36 squares. With 2 dukes, at most 34 non-duke pieces.
/// In practice, typical positions have 4-10 non-duke pieces.
const MAX_PIECES: usize = 34;

/// Stack-allocated buffer for HalfDA feature indices.
pub struct HalfDABuffer {
    pub indices: [u32; MAX_PIECES],
    pub len: usize,
}

impl HalfDABuffer {
    #[inline]
    fn new() -> Self {
        Self {
            indices: [0; MAX_PIECES],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, idx: u32) {
        debug_assert!(self.len < MAX_PIECES,
            "HalfDABuffer overflow: tried to push {} features (max {})",
            self.len + 1, MAX_PIECES);
        debug_assert!((idx as usize) < HALFDA_FEATURES,
            "HalfDA feature index {} >= {}", idx, HALFDA_FEATURES);
        self.indices[self.len] = idx;
        self.len += 1;
    }

    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        &self.indices[..self.len]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

/// Encode a game state using HalfDA features.
///
/// The encoding is from the current player's perspective:
/// - The current player's duke position is `duke_square`
/// - `piece_color` is 0 for current player's pieces, 1 for opponent's
pub fn encode_halfda(gs: &GameState) -> HalfDABuffer {
    let current_player = gs.current_player_turn();
    let board = gs.board();

    // Find current player's duke position
    let duke_coords = board
        .find(|t| t.owner == current_player && t.tile_type == TileType::Duke)
        .expect("Current player's duke must be on the board");
    let duke_square = duke_coords.y as usize * BOARD_SIZE +duke_coords.x as usize;

    let mut buf = HalfDABuffer::new();

    for (coords, placed_tile) in board.active_coordinates() {
        // Skip the current player's duke (it's the reference point)
        if placed_tile.tile_type == TileType::Duke && placed_tile.owner == current_player {
            continue;
        }

        let piece_square = coords.y as usize * BOARD_SIZE +coords.x as usize;
        let piece_type = placed_tile.tile_type.index();
        let piece_color = if placed_tile.owner == current_player { 0 } else { 1 };
        let piece_side = match placed_tile.current_side {
            CurrentSide::Initial => 0,
            CurrentSide::Flipped => 1,
        };

        let idx = duke_square * DUKE_SQUARE_STRIDE
            + piece_square * PIECE_SQUARE_STRIDE
            + piece_type * PIECE_TYPE_STRIDE
            + piece_color * PIECE_COLOR_STRIDE
            + piece_side * PIECE_SIDE_STRIDE;

        buf.push(idx as u32);
    }

    buf
}

// ── Incremental L1 Accumulator ────────────────────────────────────────────

/// Hidden size for the HalfDA accumulator.
/// Must match the hidden size used during training.
pub const HALFDA_HIDDEN: usize = 2048;

/// Incremental L1 accumulator for the HalfDA NNUE network.
///
/// Maintains the pre-ReLU hidden state (bias + sum of active embedding rows).
/// After a move, only the changed features need to be patched instead of
/// recomputing the full accumulation from scratch.
pub struct HalfDAAccumulator {
    /// Pre-ReLU hidden state: bias + weighted sum of active features. [HALFDA_HIDDEN]
    hidden: Vec<f32>,
    /// Currently active feature indices (sorted for diff computation).
    active: Vec<u32>,
}

impl Clone for HalfDAAccumulator {
    fn clone(&self) -> Self {
        Self {
            hidden: self.hidden.clone(),
            active: self.active.clone(),
        }
    }
}

impl HalfDAAccumulator {
    /// Initialize from scratch for a given position.
    ///
    /// `weights` layout: `weights[feature_idx * HALFDA_HIDDEN + neuron]`
    /// `bias`: the L1 bias vector of length `HALFDA_HIDDEN`.
    pub fn from_position(gs: &GameState, weights: &[f32], bias: &[f32]) -> Self {
        debug_assert_eq!(weights.len(), HALFDA_FEATURES * HALFDA_HIDDEN);
        debug_assert_eq!(bias.len(), HALFDA_HIDDEN);

        let buf = encode_halfda(gs);
        let mut hidden = bias.to_vec();

        for &idx in buf.as_slice() {
            let offset = idx as usize * HALFDA_HIDDEN;
            let row = &weights[offset..offset + HALFDA_HIDDEN];
            for j in 0..HALFDA_HIDDEN {
                hidden[j] += row[j];
            }
        }

        let mut active: Vec<u32> = buf.as_slice().to_vec();
        active.sort_unstable();

        Self { hidden, active }
    }

    /// Incrementally update after a move by computing the diff between old and new features.
    ///
    /// For features removed (in old but not new): subtract their embedding row.
    /// For features added (in new but not old): add their embedding row.
    ///
    /// With ~7 active features and most moves changing only 1-3 features, this
    /// touches 2-6 embedding rows instead of all 7.
    pub fn update_move(&mut self, new_features: &HalfDABuffer, weights: &[f32]) {
        let mut new_sorted: Vec<u32> = new_features.as_slice().to_vec();
        new_sorted.sort_unstable();

        // Build bitsets for O(1) membership testing
        // HALFDA_FEATURES = 67392, need 67392/32 = 2106 u32 words (or use u64)
        const WORDS: usize = (HALFDA_FEATURES + 63) / 64;
        let mut old_set = vec![0u64; WORDS];
        for &idx in &self.active {
            let i = idx as usize;
            old_set[i / 64] |= 1u64 << (i % 64);
        }
        let mut new_set = vec![0u64; WORDS];
        for &idx in &new_sorted {
            let i = idx as usize;
            new_set[i / 64] |= 1u64 << (i % 64);
        }

        // Subtract removed features (in old but not new)
        for &idx in &self.active {
            let i = idx as usize;
            if new_set[i / 64] & (1u64 << (i % 64)) == 0 {
                let offset = i * HALFDA_HIDDEN;
                let row = &weights[offset..offset + HALFDA_HIDDEN];
                for j in 0..HALFDA_HIDDEN {
                    self.hidden[j] -= row[j];
                }
            }
        }

        // Add new features (in new but not old)
        for &idx in &new_sorted {
            let i = idx as usize;
            if old_set[i / 64] & (1u64 << (i % 64)) == 0 {
                let offset = i * HALFDA_HIDDEN;
                let row = &weights[offset..offset + HALFDA_HIDDEN];
                for j in 0..HALFDA_HIDDEN {
                    self.hidden[j] += row[j];
                }
            }
        }

        self.active = new_sorted;
    }

    /// Get the current L1 output after clipped ReLU: clamp each value to [0, 1].
    pub fn output(&self) -> Vec<f32> {
        self.hidden.iter().map(|&v| v.clamp(0.0, 1.0)).collect()
    }

    /// Full forward pass through L1 (clipped ReLU) + output layer.
    ///
    /// `output_weights`: dense output layer weights, length `HALFDA_HIDDEN`.
    /// `output_bias`: scalar bias for the output neuron.
    ///
    /// Returns sigmoid(dot(clipped_relu(hidden), output_weights) + output_bias).
    pub fn evaluate(&self, output_weights: &[f32], output_bias: f32) -> f32 {
        debug_assert_eq!(output_weights.len(), HALFDA_HIDDEN);
        let mut logit = output_bias;
        for j in 0..HALFDA_HIDDEN {
            let activated = self.hidden[j].clamp(0.0, 1.0);
            logit += activated * output_weights[j];
        }
        1.0 / (1.0 + (-logit).exp())
    }
}

// ── HalfDA Evaluator ─────────────────────────────────────────────────────

use crate::game_setup::GameEvaluator;

/// Evaluator that uses a trained HalfDA NNUE network.
///
/// Implements the `GameEvaluator` trait for use in negamax search and elo tournaments.
/// Stores the L1 embedding weights/bias and the dense output layer weights/bias.
pub struct HalfDAEvaluator {
    /// L1 embedding weights: [HALFDA_FEATURES * HALFDA_HIDDEN] column-major.
    pub l1_weights: Vec<f32>,
    /// L1 bias: [HALFDA_HIDDEN].
    pub l1_bias: Vec<f32>,
    /// Output layer weights: [HALFDA_HIDDEN].
    pub output_weights: Vec<f32>,
    /// Output layer bias: scalar.
    pub output_bias: f32,
}

impl HalfDAEvaluator {
    /// Create a new HalfDAEvaluator from raw weight vectors.
    pub fn new(
        l1_weights: Vec<f32>,
        l1_bias: Vec<f32>,
        output_weights: Vec<f32>,
        output_bias: f32,
    ) -> Self {
        assert_eq!(l1_weights.len(), HALFDA_FEATURES * HALFDA_HIDDEN);
        assert_eq!(l1_bias.len(), HALFDA_HIDDEN);
        assert_eq!(output_weights.len(), HALFDA_HIDDEN);
        Self { l1_weights, l1_bias, output_weights, output_bias }
    }

    /// Load from saved checkpoint files (sparse L1 binary + dense output weights).
    pub fn load(l1_path: &str, output_weights: Vec<f32>, output_bias: f32) -> std::io::Result<Self> {
        use std::io::Read;
        let data = std::fs::read(l1_path)?;
        let mut cursor = &data[..];

        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        assert_eq!(&magic, b"HDA1", "Invalid HalfDA L1 file magic");

        let mut buf4 = [0u8; 4];
        cursor.read_exact(&mut buf4)?;
        let _version = u32::from_le_bytes(buf4);

        cursor.read_exact(&mut buf4)?;
        let num_features = u32::from_le_bytes(buf4) as usize;
        assert_eq!(num_features, HALFDA_FEATURES);

        cursor.read_exact(&mut buf4)?;
        let hidden = u32::from_le_bytes(buf4) as usize;
        assert_eq!(hidden, HALFDA_HIDDEN);

        let mut l1_bias = vec![0.0f32; hidden];
        for v in &mut l1_bias {
            cursor.read_exact(&mut buf4)?;
            *v = f32::from_le_bytes(buf4);
        }

        let mut l1_weights = vec![0.0f32; num_features * hidden];
        for v in &mut l1_weights {
            cursor.read_exact(&mut buf4)?;
            *v = f32::from_le_bytes(buf4);
        }

        Ok(Self::new(l1_weights, l1_bias, output_weights, output_bias))
    }
}

impl GameEvaluator for HalfDAEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        let acc = HalfDAAccumulator::from_position(gs, &self.l1_weights, &self.l1_bias);
        // Returns sigmoid output in [0, 1]. Map to roughly [-10, 10] for
        // compatibility with other evaluators that use raw score ranges.
        let sigmoid_val = acc.evaluate(&self.output_weights, self.output_bias);
        // Map [0,1] -> [-10, 10] linearly
        (sigmoid_val - 0.5) * 20.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duke_rust::common::coordinates::Coordinates;
    use duke_rust::game::bag::{DiscardBag, TileBag};
    use duke_rust::game::state::{GameSnapshot, GameState};
    use duke_rust::game::tile::{Owner, PlacedTile, TileType};

    /// Helper: build a GameState from a list of tiles, bags, and current player.
    fn make_state(
        tiles: Vec<(Coordinates, PlacedTile)>,
        current_turn: Owner,
    ) -> GameState {
        GameState::from_snapshot(GameSnapshot {
            tiles,
            current_turn,
            top_bag: TileBag::new(vec![]),
            bottom_bag: TileBag::new(vec![]),
            top_discard: DiscardBag::empty(),
            bottom_discard: DiscardBag::empty(),
            idle_move_count: 0,
        })
    }

    fn make_placed(owner: Owner, tile_type: TileType, flipped: bool) -> PlacedTile {
        let mut p = PlacedTile::new(owner, tile_type);
        if flipped {
            p.flip();
        }
        p
    }

    #[test]
    fn test_initial_board_feature_count() {
        // Initial board: 2 dukes + 4 footmen = 6 tiles total.
        // HalfDA encodes non-duke pieces only, so 5 features (opponent duke + 4 footmen).
        use duke_rust::game::board_setup::{DukeInitialLocation, FootmenSetup};
        let bag = TileBag::new(vec![
            TileType::Footman, TileType::Pikeman, TileType::Knight,
            TileType::Champion, TileType::Dragoon, TileType::Wizard,
            TileType::General, TileType::Marshall, TileType::Bowman,
            TileType::Longbowman, TileType::Priest,
        ]);
        let gs = GameState::new(
            &bag,
            (DukeInitialLocation::Left, FootmenSetup::Left),
            (DukeInitialLocation::Right, FootmenSetup::Right),
        );

        let features = encode_halfda(&gs);
        // 6 tiles on board, minus 1 (current player's duke) = 5 features
        assert_eq!(features.len(), 5,
            "Initial board should have 5 HalfDA features (opponent duke + 4 footmen)");
    }

    #[test]
    fn test_feature_indices_within_bounds() {
        // Create a position with several tiles
        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 1, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
            (Coordinates { x: 2, y: 3 }, make_placed(Owner::BottomPlayer, TileType::Knight, true)),
            (Coordinates { x: 4, y: 4 }, make_placed(Owner::TopPlayer, TileType::Wizard, true)),
        ];
        let gs = make_state(tiles, Owner::TopPlayer);
        let features = encode_halfda(&gs);

        for &idx in features.as_slice() {
            assert!((idx as usize) < HALFDA_FEATURES,
                "Feature index {} is out of bounds (max {})", idx, HALFDA_FEATURES);
        }
    }

    #[test]
    fn test_no_duplicate_indices() {
        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 1, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
            (Coordinates { x: 2, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, true)),
            (Coordinates { x: 3, y: 3 }, make_placed(Owner::BottomPlayer, TileType::Knight, false)),
            (Coordinates { x: 4, y: 4 }, make_placed(Owner::BottomPlayer, TileType::Pikeman, true)),
        ];
        let gs = make_state(tiles, Owner::TopPlayer);
        let features = encode_halfda(&gs);

        let mut seen = std::collections::HashSet::new();
        for &idx in features.as_slice() {
            assert!(seen.insert(idx), "Duplicate HalfDA feature index: {}", idx);
        }
    }

    #[test]
    fn test_encoding_flips_with_player_perspective() {
        // Same board, but different current player -> different encoding
        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 2, y: 2 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
        ];

        let gs_top = make_state(tiles.clone(), Owner::TopPlayer);
        let gs_bottom = make_state(tiles, Owner::BottomPlayer);

        let feat_top = encode_halfda(&gs_top);
        let feat_bottom = encode_halfda(&gs_bottom);

        // When TopPlayer is current: duke at (0,0), encode bottom duke + footman
        // When BottomPlayer is current: duke at (5,5), encode top duke + footman
        // The features must differ because duke_square and piece_color both change
        let top_set: std::collections::HashSet<u32> = feat_top.as_slice().iter().copied().collect();
        let bottom_set: std::collections::HashSet<u32> = feat_bottom.as_slice().iter().copied().collect();
        assert_ne!(top_set, bottom_set,
            "HalfDA encoding should differ when current player changes");
    }

    #[test]
    fn test_known_feature_index() {
        // Manually compute a known feature index:
        // Duke at (0,0) = square 0
        // Footman at (1,0) = square 1, type=1 (Footman), color=0 (mine), side=0 (Initial)
        // idx = 0 * 1872 + 1 * 52 + 1 * 4 + 0 * 2 + 0 = 56
        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 1, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
        ];
        let gs = make_state(tiles, Owner::TopPlayer);
        let features = encode_halfda(&gs);

        // Should have 2 features: opponent duke and my footman
        assert_eq!(features.len(), 2);

        let indices: Vec<u32> = features.as_slice().to_vec();
        // My footman: duke_sq=0, piece_sq=1, type=1, color=0, side=0
        // idx = 0*1872 + 1*52 + 1*4 + 0*2 + 0 = 56
        assert!(indices.contains(&56),
            "Expected feature index 56 for own Footman at (1,0), got {:?}", indices);

        // Opponent duke: duke_sq=0, piece_sq=35, type=0(Duke), color=1(opp), side=0(Initial)
        // idx = 0*1872 + 35*52 + 0*4 + 1*2 + 0 = 1822
        assert!(indices.contains(&1822),
            "Expected feature index 1822 for opponent Duke at (5,5), got {:?}", indices);
    }

    // ── Accumulator tests ────────────────────────────────────────────────

    /// Create small random weights for testing (use a small hidden size for speed).
    fn make_test_weights(hidden: usize) -> (Vec<f32>, Vec<f32>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = 0.1f32;
        let weights: Vec<f32> = (0..HALFDA_FEATURES * hidden)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        let bias: Vec<f32> = (0..hidden)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        (weights, bias)
    }

    #[test]
    fn test_accumulator_matches_from_scratch() {
        // Build a position and verify the accumulator hidden state matches
        // manual computation.
        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 1, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
            (Coordinates { x: 3, y: 3 }, make_placed(Owner::BottomPlayer, TileType::Knight, true)),
        ];
        let gs = make_state(tiles, Owner::TopPlayer);

        let (weights, bias) = make_test_weights(HALFDA_HIDDEN);
        let acc = HalfDAAccumulator::from_position(&gs, &weights, &bias);

        // Manually compute expected hidden state
        let features = encode_halfda(&gs);
        let mut expected = bias.clone();
        for &idx in features.as_slice() {
            let offset = idx as usize * HALFDA_HIDDEN;
            for j in 0..HALFDA_HIDDEN {
                expected[j] += weights[offset + j];
            }
        }

        for j in 0..HALFDA_HIDDEN {
            assert!(
                (acc.hidden[j] - expected[j]).abs() < 1e-6,
                "hidden[{}] mismatch: acc={}, expected={}", j, acc.hidden[j], expected[j]
            );
        }
    }

    #[test]
    fn test_accumulator_update_matches_recompute() {
        // Position 1: before move
        let tiles1 = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 1, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
            (Coordinates { x: 3, y: 3 }, make_placed(Owner::BottomPlayer, TileType::Knight, true)),
        ];
        // Position 2: after move (footman moved from (1,0) to (2,1))
        let tiles2 = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 2, y: 1 }, make_placed(Owner::TopPlayer, TileType::Footman, true)), // flipped after move
            (Coordinates { x: 3, y: 3 }, make_placed(Owner::BottomPlayer, TileType::Knight, true)),
        ];

        let gs1 = make_state(tiles1, Owner::TopPlayer);
        let gs2 = make_state(tiles2, Owner::TopPlayer);

        let (weights, bias) = make_test_weights(HALFDA_HIDDEN);

        // Method 1: from-scratch computation for position 2
        let acc_scratch = HalfDAAccumulator::from_position(&gs2, &weights, &bias);

        // Method 2: incremental update from position 1
        let mut acc_incr = HalfDAAccumulator::from_position(&gs1, &weights, &bias);
        let new_features = encode_halfda(&gs2);
        acc_incr.update_move(&new_features, &weights);

        // They must match exactly (no floating point drift since operations are identical)
        for j in 0..HALFDA_HIDDEN {
            assert!(
                (acc_incr.hidden[j] - acc_scratch.hidden[j]).abs() < 1e-5,
                "After update, hidden[{}] mismatch: incremental={}, scratch={}",
                j, acc_incr.hidden[j], acc_scratch.hidden[j]
            );
        }
    }

    #[test]
    fn test_accumulator_update_with_capture() {
        // Position 1: before capture
        let tiles1 = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 1, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
            (Coordinates { x: 2, y: 1 }, make_placed(Owner::BottomPlayer, TileType::Knight, true)),
        ];
        // Position 2: after capture (footman captures knight at (2,1))
        let tiles2 = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 2, y: 1 }, make_placed(Owner::TopPlayer, TileType::Footman, true)),
        ];

        let gs1 = make_state(tiles1, Owner::TopPlayer);
        let gs2 = make_state(tiles2, Owner::TopPlayer);

        let (weights, bias) = make_test_weights(HALFDA_HIDDEN);

        let acc_scratch = HalfDAAccumulator::from_position(&gs2, &weights, &bias);
        let mut acc_incr = HalfDAAccumulator::from_position(&gs1, &weights, &bias);
        let new_features = encode_halfda(&gs2);
        acc_incr.update_move(&new_features, &weights);

        for j in 0..HALFDA_HIDDEN {
            assert!(
                (acc_incr.hidden[j] - acc_scratch.hidden[j]).abs() < 1e-5,
                "After capture update, hidden[{}] mismatch: incremental={}, scratch={}",
                j, acc_incr.hidden[j], acc_scratch.hidden[j]
            );
        }
    }

    #[test]
    fn test_accumulator_evaluate_range() {
        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 1, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
        ];
        let gs = make_state(tiles, Owner::TopPlayer);

        let (weights, bias) = make_test_weights(HALFDA_HIDDEN);
        let acc = HalfDAAccumulator::from_position(&gs, &weights, &bias);

        // Output weights for evaluation
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let output_weights: Vec<f32> = (0..HALFDA_HIDDEN)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        let output_bias = 0.0f32;

        let result = acc.evaluate(&output_weights, output_bias);
        // Sigmoid output must be in (0, 1)
        assert!(result > 0.0 && result < 1.0,
            "Accumulator evaluate should return sigmoid in (0,1), got {}", result);
    }

    #[test]
    fn test_halfda_evaluator_returns_bounded_values() {
        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 1, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
        ];
        let gs = make_state(tiles, Owner::TopPlayer);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let l1_weights: Vec<f32> = (0..HALFDA_FEATURES * HALFDA_HIDDEN)
            .map(|_| rng.gen_range(-0.01..0.01))
            .collect();
        let l1_bias = vec![0.0f32; HALFDA_HIDDEN];
        let output_weights: Vec<f32> = (0..HALFDA_HIDDEN)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        let output_bias = 0.0f32;

        let evaluator = HalfDAEvaluator::new(l1_weights, l1_bias, output_weights, output_bias);
        let score = evaluator.evaluate(&gs);
        // HalfDAEvaluator maps sigmoid [0,1] -> [-10, 10]
        assert!(score >= -10.0 && score <= 10.0,
            "HalfDAEvaluator score should be in [-10, 10], got {}", score);
    }

    #[test]
    fn test_accumulator_output_clipped_relu() {
        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 1, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
        ];
        let gs = make_state(tiles, Owner::TopPlayer);

        let (weights, bias) = make_test_weights(HALFDA_HIDDEN);
        let acc = HalfDAAccumulator::from_position(&gs, &weights, &bias);
        let output = acc.output();

        // All values should be clamped to [0, 1]
        for (j, &v) in output.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0,
                "output[{}] = {} is outside [0, 1]", j, v);
        }
    }
}
