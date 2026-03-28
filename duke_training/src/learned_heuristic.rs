//! Learned heuristic evaluator using polynomial feature expansion + ridge regression.
//!
//! 24 features total:
//! - 4 base heuristic differences (approx, for speed)
//! - 14 polynomial expansion terms (quadratic + cross + cubic)
//! - 5 new cheap features
//! - 1 bias term

use std::fs;

use duke_rust::game::ai::heuristics::{Heuristic, Heuristics};
use duke_rust::game::board::PossibleMove;
use duke_rust::game::state::GameState;
use duke_rust::game::tile::TileType;
use strum::EnumCount;

use crate::game_setup::GameEvaluator;

/// Total number of features in the full polynomial expansion.
pub const NUM_FEATURES: usize = 24;

/// Center squares on the 6x6 board: (2,2), (3,2), (2,3), (3,3).
const CENTER_SQUARES: [(u8, u8); 4] = [(2, 2), (3, 2), (2, 3), (3, 3)];

/// Learned weight vector for the polynomial heuristic.
#[derive(Debug, Clone)]
pub struct LearnedHeuristicWeights {
    pub weights: [f64; NUM_FEATURES],
}

impl Default for LearnedHeuristicWeights {
    fn default() -> Self {
        Self {
            weights: [0.0; NUM_FEATURES],
        }
    }
}

/// Extract the 24-dimensional feature vector from a game state.
///
/// Features are computed from the perspective of `gs.current_player_turn()`.
pub fn extract_features(gs: &GameState) -> [f64; NUM_FEATURES] {
    let owner = gs.current_player_turn();
    let opp = owner.next_player();

    // Per-player heuristic values (computed once, reused for differences and ratios)
    let my_duke_mob = Heuristics::DukeMovementOptions.approx_evaluate_for_owner(owner, gs);
    let opp_duke_mob = Heuristics::DukeMovementOptions.approx_evaluate_for_owner(opp, gs);
    let my_total_mob = Heuristics::TotalMovementOptions.approx_evaluate_for_owner(owner, gs);
    let opp_total_mob = Heuristics::TotalMovementOptions.approx_evaluate_for_owner(opp, gs);

    // Base heuristic differences
    let x1 = my_duke_mob - opp_duke_mob;
    let x2 = Heuristics::TotalTilesOnBoard.approx_difference(owner, gs);
    let x3 = my_total_mob - opp_total_mob;
    let x4 = Heuristics::DiscardedUnits.approx_difference(owner, gs);

    // x5: duke_guard_diff
    let my_guard = if gs.is_duke_in_guard(owner) { 1.0 } else { 0.0 };
    let opp_guard = if gs.is_duke_in_guard(opp) { 1.0 } else { 0.0 };
    let x5 = opp_guard - my_guard;

    // x6: bag_emptiness_diff
    let my_bag = gs.bag_for_owner(owner).remaining().len() as f64;
    let opp_bag = gs.bag_for_owner(opp).remaining().len() as f64;
    let x6 = opp_bag - my_bag;

    // Fetch tiles once per owner (avoids 4 separate Vec allocations)
    let my_tiles = gs.get_tiles_for_owner(owner);
    let opp_tiles = gs.get_tiles_for_owner(opp);

    // x7: tile_adjacency_diff
    let my_adj = count_adjacent_pairs_from(&my_tiles) as f64;
    let opp_adj = count_adjacent_pairs_from(&opp_tiles) as f64;
    let x7 = my_adj - opp_adj;

    // x8: center_control_diff
    let my_center = count_center_tiles_from(&my_tiles) as f64;
    let opp_center = count_center_tiles_from(&opp_tiles) as f64;
    let x8 = my_center - opp_center;

    // x9: duke_mobility_ratio_diff
    let my_ratio = my_duke_mob / my_total_mob.max(1.0);
    let opp_ratio = opp_duke_mob / opp_total_mob.max(1.0);
    let x9 = my_ratio - opp_ratio;

    // Build the full 24-feature vector
    let mut features = [0.0f64; NUM_FEATURES];

    // Base (0..4)
    features[0] = x1;
    features[1] = x2;
    features[2] = x3;
    features[3] = x4;

    // Quadratic (4..8): x1^2, x2^2, x3^2, x4^2
    features[4] = x1 * x1;
    features[5] = x2 * x2;
    features[6] = x3 * x3;
    features[7] = x4 * x4;

    // Cross terms (8..14): x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4
    features[8] = x1 * x2;
    features[9] = x1 * x3;
    features[10] = x1 * x4;
    features[11] = x2 * x3;
    features[12] = x2 * x4;
    features[13] = x3 * x4;

    // Cubic (14..18): x1^3, x2^3, x3^3, x4^3
    features[14] = x1 * x1 * x1;
    features[15] = x2 * x2 * x2;
    features[16] = x3 * x3 * x3;
    features[17] = x4 * x4 * x4;

    // New cheap features (18..23)
    features[18] = x5;
    features[19] = x6;
    features[20] = x7;
    features[21] = x8;
    features[22] = x9;

    // Bias (23)
    features[23] = 1.0;

    features
}

/// Count orthogonally adjacent own-tile pairs from pre-fetched tile list.
fn count_adjacent_pairs_from(tiles: &[(duke_rust::common::coordinates::Coordinates, &duke_rust::game::tile::PlacedTile)]) -> usize {
    let mut count = 0;
    for i in 0..tiles.len() {
        for j in (i + 1)..tiles.len() {
            let (c1, _) = &tiles[i];
            let (c2, _) = &tiles[j];
            // Orthogonally adjacent: Manhattan distance == 1
            let dx = (c1.x as i32 - c2.x as i32).unsigned_abs();
            let dy = (c1.y as i32 - c2.y as i32).unsigned_abs();
            if (dx == 1 && dy == 0) || (dx == 0 && dy == 1) {
                count += 1;
            }
        }
    }
    count
}

/// Count tiles in the center 4 squares from pre-fetched tile list.
fn count_center_tiles_from(tiles: &[(duke_rust::common::coordinates::Coordinates, &duke_rust::game::tile::PlacedTile)]) -> usize {
    tiles.iter()
        .filter(|(c, _)| CENTER_SQUARES.contains(&(c.x, c.y)))
        .count()
}

/// Compute 4 Manhattan-distance proximity features from the perspective of
/// `gs.current_player_turn()`:
///
/// - `[0]` my_units_near_my_duke: count of my non-duke tiles within Manhattan distance 2 of my duke
/// - `[1]` enemy_units_near_my_duke: count of enemy tiles within Manhattan distance 2 of my duke
/// - `[2]` my_units_near_enemy_duke: count of my tiles within Manhattan distance 2 of enemy duke
/// - `[3]` enemy_units_near_enemy_duke: count of enemy non-duke tiles within Manhattan distance 2 of enemy duke
pub fn manhattan_distance_features(gs: &GameState) -> [f64; 4] {
    let me = gs.current_player_turn();
    let opp = me.next_player();

    let my_duke = gs.duke_coordinate(me);
    let enemy_duke = gs.duke_coordinate(opp);

    let mut counts = [0.0f64; 4];

    for (coords, tile) in gs.board().active_coordinates() {
        let is_mine = tile.owner == me;
        let is_duke_tile = tile.tile_type.is_duke();

        let dx_my = (coords.x as i32 - my_duke.x as i32).unsigned_abs();
        let dy_my = (coords.y as i32 - my_duke.y as i32).unsigned_abs();
        let dist_my = dx_my + dy_my;

        let dx_en = (coords.x as i32 - enemy_duke.x as i32).unsigned_abs();
        let dy_en = (coords.y as i32 - enemy_duke.y as i32).unsigned_abs();
        let dist_en = dx_en + dy_en;

        if dist_my <= 2 {
            if is_mine && !is_duke_tile {
                counts[0] += 1.0; // my non-duke near my duke
            } else if !is_mine {
                counts[1] += 1.0; // enemy near my duke
            }
        }

        if dist_en <= 2 {
            if is_mine && !is_duke_tile {
                counts[2] += 1.0; // my non-duke near enemy duke
            } else if !is_mine && !is_duke_tile {
                counts[3] += 1.0; // enemy non-duke near enemy duke
            }
        }
    }

    counts
}

/// Per-tile-type discard counts for both players.
/// Returns 26 values: [my_duke_discards, my_footman_discards, ..., opp_duke_discards, opp_footman_discards, ...]
/// 13 tile types × 2 players.
pub fn discard_vector(gs: &GameState) -> [f64; 26] {
    let me = gs.current_player_turn();
    let opp = me.next_player();

    let mut counts = [0.0f64; 26];

    for tile in gs.discard_bag_for(me).existing() {
        counts[tile.index()] += 1.0;
    }
    for tile in gs.discard_bag_for(opp).existing() {
        counts[TileType::COUNT + tile.index()] += 1.0;
    }

    counts
}

/// Total feature count for the new combined feature set.
/// Manhattan(4) + board_control(9) + duke_mobility(2) + discard_vector(26) = 41
pub const NUM_COMBINED_FEATURES: usize = 41;

/// Extract the combined feature set: all new features, per-player, no guard checking.
///
/// Uses `board_control_features_with_duke_mob` to compute board-control and
/// duke-mobility in a single move-generation pass per player, avoiding the
/// redundant second pass that `duke_mobility_no_guard` would perform.
pub fn extract_combined_features(gs: &GameState) -> [f64; NUM_COMBINED_FEATURES] {
    let manhattan = manhattan_distance_features(gs);
    let (control, duke_mob) = board_control_features_with_duke_mob(gs);
    let discards = discard_vector(gs);

    let mut f = [0.0f64; NUM_COMBINED_FEATURES];
    f[0..4].copy_from_slice(&manhattan);
    f[4..13].copy_from_slice(&control);
    f[13..15].copy_from_slice(&duke_mob);
    f[15..41].copy_from_slice(&discards);
    f
}

/// Learned weight vector for the combined 41-feature set.
#[derive(Debug, Clone)]
pub struct CombinedWeights {
    pub weights: [f64; NUM_COMBINED_FEATURES],
}

impl Default for CombinedWeights {
    fn default() -> Self {
        Self { weights: [0.0; NUM_COMBINED_FEATURES] }
    }
}

impl CombinedWeights {
    pub fn evaluate_raw(&self, gs: &GameState) -> f64 {
        let features = extract_combined_features(gs);
        let mut score = 0.0;
        for i in 0..NUM_COMBINED_FEATURES {
            score += self.weights[i] * features[i];
        }
        score
    }

    /// Save weights to a JSON file (same format as LearnedHeuristicWeights).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = format!(
            "{{\"weights\":[{}]}}",
            self.weights
                .iter()
                .map(|w| format!("{:.15e}", w))
                .collect::<Vec<_>>()
                .join(",")
        );
        fs::write(path, json)
    }

    /// Load weights from a JSON file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let data = fs::read_to_string(path)?;
        let start = data.find('[').ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "No '[' found in JSON")
        })?;
        let end = data.find(']').ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "No ']' found in JSON")
        })?;
        let array_str = &data[start + 1..end];
        let values: Vec<f64> = array_str
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to parse weight: {}", e),
                )
            })?;
        if values.len() != NUM_COMBINED_FEATURES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Expected {} weights, found {}",
                    NUM_COMBINED_FEATURES,
                    values.len()
                ),
            ));
        }
        let mut weights = [0.0f64; NUM_COMBINED_FEATURES];
        weights.copy_from_slice(&values);
        Ok(Self { weights })
    }
}

impl GameEvaluator for CombinedWeights {
    fn evaluate(&self, gs: &GameState) -> f32 {
        self.evaluate_raw(gs) as f32
    }
}

/// Board control features plus duke mobility, computed in a single pass over
/// each player's moves.
///
/// Returns `(control, duke_mob)` where:
///
/// `control` — 9 values:
///   [0] my_approx_moves
///   [1] opp_approx_moves
///   [2] my_reachable_squares
///   [3] opp_reachable_squares
///   [4] contested_squares
///   [5] my_defended   — how many of my tiles sit on squares I can reach
///   [6] my_threatened — how many enemy tiles sit on squares I can reach
///   [7] opp_defended  — how many of opponent's tiles sit on squares opponent can reach
///   [8] opp_threatened — how many of my tiles sit on squares opponent can reach
///
/// `duke_mob` — 2 values:
///   [0] my_duke_mobility   (move count for owner's duke, ignoring guard)
///   [1] opp_duke_mobility  (move count for opponent's duke, ignoring guard)
///
/// Moves are computed while ignoring the guard constraint (the expensive part),
/// making this a cheap approximation. Only tile-move destinations (not placements)
/// contribute to the reachable-squares arrays.
pub fn board_control_features_with_duke_mob(gs: &GameState) -> ([f64; 9], [f64; 2]) {
    let owner = gs.current_player_turn();
    let opp = owner.next_player();

    let my_duke_coord = gs.duke_coordinate(owner);
    let opp_duke_coord = gs.duke_coordinate(opp);

    let mut my_reach = [false; 36];
    let mut opp_reach = [false; 36];

    // Only count tile-movement moves (not placements) so that approx_moves
    // is consistent with reachable_squares — both measure on-board tile actions.
    let mut my_approx_moves = 0u32;
    let mut my_duke_moves = 0u32;
    for pm in gs.all_valid_game_moves_for_ignoring_guard(owner) {
        if let PossibleMove::ApplyNonCommandTileAction { src, dst, .. } = &pm {
            my_approx_moves += 1;
            if *src == my_duke_coord {
                my_duke_moves += 1;
            }
            let idx = dst.y as usize * 6 + dst.x as usize;
            debug_assert!(idx < 36, "move destination ({}, {}) maps to index {} outside 6x6 board", dst.x, dst.y, idx);
            my_reach[idx] = true;
        }
    }

    let mut opp_approx_moves = 0u32;
    let mut opp_duke_moves = 0u32;
    for pm in gs.all_valid_game_moves_for_ignoring_guard(opp) {
        if let PossibleMove::ApplyNonCommandTileAction { src, dst, .. } = &pm {
            opp_approx_moves += 1;
            if *src == opp_duke_coord {
                opp_duke_moves += 1;
            }
            let idx = dst.y as usize * 6 + dst.x as usize;
            debug_assert!(idx < 36, "move destination ({}, {}) maps to index {} outside 6x6 board", dst.x, dst.y, idx);
            opp_reach[idx] = true;
        }
    }

    let mut my_reachable = 0u32;
    let mut opp_reachable = 0u32;
    let mut contested = 0u32;
    for i in 0..36 {
        if my_reach[i] {
            my_reachable += 1;
        }
        if opp_reach[i] {
            opp_reachable += 1;
        }
        if my_reach[i] && opp_reach[i] {
            contested += 1;
        }
    }

    // Defended/threatened: iterate over all tiles on the board
    let mut my_defended = 0u32;
    let mut my_threatened = 0u32;
    let mut opp_defended = 0u32;
    let mut opp_threatened = 0u32;
    for (coords, tile) in gs.board().active_coordinates() {
        let idx = coords.y as usize * 6 + coords.x as usize;
        let is_mine = tile.owner == owner;
        if is_mine {
            if my_reach[idx] {
                my_defended += 1;
            }
            if opp_reach[idx] {
                opp_threatened += 1;
            }
        } else {
            if my_reach[idx] {
                my_threatened += 1;
            }
            if opp_reach[idx] {
                opp_defended += 1;
            }
        }
    }

    (
        [
            my_approx_moves as f64,
            opp_approx_moves as f64,
            my_reachable as f64,
            opp_reachable as f64,
            contested as f64,
            my_defended as f64,
            my_threatened as f64,
            opp_defended as f64,
            opp_threatened as f64,
        ],
        [
            my_duke_moves as f64,
            opp_duke_moves as f64,
        ],
    )
}

impl LearnedHeuristicWeights {
    /// Evaluate a game state, returning a raw score (higher = better for current player).
    pub fn evaluate_raw(&self, gs: &GameState) -> f64 {
        let features = extract_features(gs);
        let mut score = 0.0;
        for i in 0..NUM_FEATURES {
            score += self.weights[i] * features[i];
        }
        score
    }

    /// Save weights to a JSON file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = format!(
            "{{\"weights\":[{}]}}",
            self.weights
                .iter()
                .map(|w| format!("{:.15e}", w))
                .collect::<Vec<_>>()
                .join(",")
        );
        fs::write(path, json)
    }

    /// Load weights from a JSON file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let data = fs::read_to_string(path)?;
        // Minimal JSON parser: find the array between [ and ]
        let start = data.find('[').ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "No '[' found in JSON")
        })?;
        let end = data.find(']').ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "No ']' found in JSON")
        })?;
        let array_str = &data[start + 1..end];
        let values: Vec<f64> = array_str
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to parse weight: {}", e),
                )
            })?;
        if values.len() != NUM_FEATURES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Expected {} weights, found {}",
                    NUM_FEATURES,
                    values.len()
                ),
            ));
        }
        let mut weights = [0.0f64; NUM_FEATURES];
        weights.copy_from_slice(&values);
        Ok(Self { weights })
    }
}

impl GameEvaluator for LearnedHeuristicWeights {
    fn evaluate(&self, gs: &GameState) -> f32 {
        self.evaluate_raw(gs) as f32
    }
}

/// Total feature count for the merged feature set: 24 expensive + 41 combined = 65.
pub const NUM_ALL_FEATURES: usize = NUM_FEATURES + NUM_COMBINED_FEATURES;

/// Learned weight vector for the merged 65-feature set (24 expensive + 41 combined).
#[derive(Debug, Clone)]
pub struct AllFeaturesWeights {
    pub weights: [f64; NUM_ALL_FEATURES],
}

impl Default for AllFeaturesWeights {
    fn default() -> Self {
        Self { weights: [0.0; NUM_ALL_FEATURES] }
    }
}

impl AllFeaturesWeights {
    pub fn evaluate_raw(&self, gs: &GameState) -> f64 {
        let expensive = extract_features(gs);
        let combined = extract_combined_features(gs);
        let mut score = 0.0;
        for i in 0..NUM_FEATURES {
            score += self.weights[i] * expensive[i];
        }
        for i in 0..NUM_COMBINED_FEATURES {
            score += self.weights[NUM_FEATURES + i] * combined[i];
        }
        score
    }

    /// Save weights to a JSON file (same format as other weight structs).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = format!(
            "{{\"weights\":[{}]}}",
            self.weights
                .iter()
                .map(|w| format!("{:.15e}", w))
                .collect::<Vec<_>>()
                .join(",")
        );
        fs::write(path, json)
    }

    /// Load weights from a JSON file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let data = fs::read_to_string(path)?;
        let start = data.find('[').ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "No '[' found in JSON")
        })?;
        let end = data.find(']').ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "No ']' found in JSON")
        })?;
        let array_str = &data[start + 1..end];
        let values: Vec<f64> = array_str
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to parse weight: {}", e),
                )
            })?;
        if values.len() != NUM_ALL_FEATURES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Expected {} weights, found {}",
                    NUM_ALL_FEATURES,
                    values.len()
                ),
            ));
        }
        let mut weights = [0.0f64; NUM_ALL_FEATURES];
        weights.copy_from_slice(&values);
        Ok(Self { weights })
    }
}

impl GameEvaluator for AllFeaturesWeights {
    fn evaluate(&self, gs: &GameState) -> f32 {
        self.evaluate_raw(gs) as f32
    }
}

/// Load a JSON weight file and return the raw weight vector.
///
/// Both `LearnedHeuristicWeights` (24) and `CombinedWeights` (41) use the same
/// `{"weights": [...]}` format. This function parses the file and returns the
/// raw `Vec<f64>` so the caller can dispatch by length.
pub fn load_lr_weights_raw(path: &str) -> std::io::Result<Vec<f64>> {
    let data = fs::read_to_string(path)?;
    let start = data.find('[').ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "No '[' found in JSON")
    })?;
    let end = data.find(']').ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "No ']' found in JSON")
    })?;
    let array_str = &data[start + 1..end];
    array_str
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse weight: {}", e),
            )
        })
}

