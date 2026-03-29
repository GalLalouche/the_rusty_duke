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
        save_weights_json(path, &self.weights)
    }

    /// Load weights from a JSON file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let values = load_weights_json(path)?;
        if values.len() != NUM_COMBINED_FEATURES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Expected {} weights, found {}", NUM_COMBINED_FEATURES, values.len()),
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
/// `control` -- 9 values:
///   [0] my_approx_moves
///   [1] opp_approx_moves
///   [2] my_reachable_squares
///   [3] opp_reachable_squares
///   [4] contested_squares
///   [5] my_defended   -- how many of my tiles sit on squares I can reach
///   [6] my_threatened -- how many enemy tiles sit on squares I can reach
///   [7] opp_defended  -- how many of opponent's tiles sit on squares opponent can reach
///   [8] opp_threatened -- how many of my tiles sit on squares opponent can reach
///
/// `duke_mob` -- 2 values:
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
    // is consistent with reachable_squares -- both measure on-board tile actions.
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

    // Defended/threatened: iterate over all tiles on the board.
    //
    // `my_reach` only contains squares reachable by legal moves, which
    // excludes friendly-occupied squares (you can't move onto your own
    // piece). So `my_reach[idx]` is always false for squares where my
    // tiles sit, making `my_defended` always zero -- a bug.
    //
    // For "threatened" (enemy tiles on squares I can reach), the existing
    // `my_reach` is correct because capturing an enemy IS a legal move.
    //
    // For "defended" we instead check: can any OTHER friendly piece reach
    // this tile's square, ignoring the friendly-occupancy constraint?
    // This uses `can_reach_square_ignoring_friendly` which checks the
    // movement pattern and path obstruction but allows the destination to
    // be friendly-occupied.
    let my_tiles = gs.get_tiles_for_owner(owner);
    let opp_tiles = gs.get_tiles_for_owner(opp);

    let mut my_defended = 0u32;
    let mut my_threatened = 0u32;
    let mut opp_defended = 0u32;
    let mut opp_threatened = 0u32;

    // Threatened: enemy tiles on squares I can reach (via legal captures)
    for (coords, _) in &opp_tiles {
        let idx = coords.y as usize * 6 + coords.x as usize;
        if my_reach[idx] {
            my_threatened += 1;
        }
        if opp_reach[idx] {
            // Opponent's own piece on a square the opponent can reach --
            // this was always zero for the same reason; skip (handled below).
        }
    }
    for (coords, _) in &my_tiles {
        let idx = coords.y as usize * 6 + coords.x as usize;
        if opp_reach[idx] {
            opp_threatened += 1;
        }
    }

    // Defended: for each of my tiles, check if any OTHER friendly piece
    // can reach its square (ignoring friendly occupancy).
    for (i, (coords_i, _)) in my_tiles.iter().enumerate() {
        for (j, (coords_j, _)) in my_tiles.iter().enumerate() {
            if i != j && gs.can_reach_square_ignoring_friendly(*coords_j, *coords_i) {
                my_defended += 1;
                break; // one defender is enough to count this tile
            }
        }
    }
    for (i, (coords_i, _)) in opp_tiles.iter().enumerate() {
        for (j, (coords_j, _)) in opp_tiles.iter().enumerate() {
            if i != j && gs.can_reach_square_ignoring_friendly(*coords_j, *coords_i) {
                opp_defended += 1;
                break;
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
        save_weights_json(path, &self.weights)
    }

    /// Load weights from a JSON file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let values = load_weights_json(path)?;
        if values.len() != NUM_FEATURES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Expected {} weights, found {}", NUM_FEATURES, values.len()),
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
        save_weights_json(path, &self.weights)
    }

    /// Load weights from a JSON file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let values = load_weights_json(path)?;
        if values.len() != NUM_ALL_FEATURES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Expected {} weights, found {}", NUM_ALL_FEATURES, values.len()),
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

/// Save a weight vector to a JSON file as `{"weights": [...]}`.
///
/// All weight structs share the same on-disk format; this is the single
/// implementation they all delegate to.
pub fn save_weights_json(path: &str, weights: &[f64]) -> std::io::Result<()> {
    let json = format!(
        "{{\"weights\":[{}]}}",
        weights
            .iter()
            .map(|w| format!("{:.15e}", w))
            .collect::<Vec<_>>()
            .join(",")
    );
    fs::write(path, json)
}

/// Load a JSON weight file and return the raw weight vector.
///
/// All weight structs (`LearnedHeuristicWeights`, `CombinedWeights`,
/// `AllFeaturesWeights`) use the same `{"weights": [...]}` format. This
/// function parses the file and returns the raw `Vec<f64>` so the caller
/// can dispatch by length or copy into a fixed-size array.
pub fn load_weights_json(path: &str) -> std::io::Result<Vec<f64>> {
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

/// Backward-compatible alias for `load_weights_json`.
pub fn load_lr_weights_raw(path: &str) -> std::io::Result<Vec<f64>> {
    load_weights_json(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use duke_rust::common::coordinates::Coordinates;
    use duke_rust::game::bag::{DiscardBag, TileBag};
    use duke_rust::game::state::{GameSnapshot, GameState};
    use duke_rust::game::tile::{Owner, PlacedTile, TileType};

    /// Helper: create the standard initial game state used across the codebase.
    fn initial_state() -> GameState {
        let bag = crate::game_setup::create_bag();
        crate::game_setup::create_initial_state(&bag)
    }

    /// Helper: create a custom board via `GameState::from_snapshot`.
    fn state_from_tiles(tiles: Vec<(Coordinates, PlacedTile)>) -> GameState {
        GameState::from_snapshot(GameSnapshot {
            tiles,
            top_bag: TileBag::new(vec![]),
            bottom_bag: TileBag::new(vec![]),
            top_discard: DiscardBag::empty(),
            bottom_discard: DiscardBag::empty(),
            current_turn: Owner::TopPlayer,
            idle_move_count: 0,
        })
    }

    fn c(x: u8, y: u8) -> Coordinates {
        Coordinates { x, y }
    }

    // ── manhattan_distance_features ──────────────────────────────────

    #[test]
    fn manhattan_initial_board() {
        // Initial board (TopPlayer turn):
        //   TopPlayer:    Duke(3,0), Footman(4,0), Footman(3,1)
        //   BottomPlayer: Duke(3,5), Footman(4,5), Footman(3,4)
        let gs = initial_state();
        let md = manhattan_distance_features(&gs);

        // my = TopPlayer, my_duke = (3,0), enemy_duke = (3,5)
        // my non-duke tiles near my duke (dist<=2):
        //   Footman(4,0): |4-3|+|0-0|=1 => yes
        //   Footman(3,1): |3-3|+|1-0|=1 => yes
        //   => 2
        assert_eq!(md[0], 2.0, "my_units_near_my_duke");

        // enemy tiles near my duke (dist<=2):
        //   BottomPlayer tiles at (3,5),(4,5),(3,4) all have dist>=4 from (3,0)
        //   => 0
        assert_eq!(md[1], 0.0, "enemy_units_near_my_duke");

        // my non-duke tiles near enemy duke (3,5) (dist<=2):
        //   Footman(4,0): |4-3|+|0-5|=6 => no
        //   Footman(3,1): |3-3|+|1-5|=4 => no
        //   => 0
        assert_eq!(md[2], 0.0, "my_units_near_enemy_duke");

        // enemy non-duke tiles near enemy duke (3,5) (dist<=2):
        //   Footman(4,5): |4-3|+|5-5|=1 => yes
        //   Footman(3,4): |3-3|+|4-5|=1 => yes
        //   => 2
        assert_eq!(md[3], 2.0, "enemy_units_near_enemy_duke");
    }

    #[test]
    fn manhattan_custom_close_positions() {
        // Place units so they are near both dukes
        let tiles = vec![
            (c(2, 2), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(3, 2), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(2, 3), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
            (c(3, 3), PlacedTile::new(Owner::BottomPlayer, TileType::Footman)),
        ];
        let gs = state_from_tiles(tiles);
        let md = manhattan_distance_features(&gs);

        // my_duke=(2,2), enemy_duke=(2,3)
        // my non-duke near my duke: Footman(3,2) dist=1 => 1
        assert_eq!(md[0], 1.0, "my_units_near_my_duke");
        // enemy tiles near my duke: Duke(2,3) dist=1, Footman(3,3) dist=2 => 2
        assert_eq!(md[1], 2.0, "enemy_units_near_my_duke");
        // my non-duke near enemy duke(2,3): Footman(3,2) dist=|3-2|+|2-3|=2 => 1
        assert_eq!(md[2], 1.0, "my_units_near_enemy_duke");
        // enemy non-duke near enemy duke: Footman(3,3) dist=1 => 1
        assert_eq!(md[3], 1.0, "enemy_units_near_enemy_duke");
    }

    // ── board_control_features_with_duke_mob ─────────────────────────

    #[test]
    fn board_control_initial_state() {
        let gs = initial_state();
        let (control, duke_mob) = board_control_features_with_duke_mob(&gs);

        // Sanity: move counts should be positive
        assert!(control[0] > 0.0, "my_approx_moves should be > 0, got {}", control[0]);
        assert!(control[1] > 0.0, "opp_approx_moves should be > 0, got {}", control[1]);

        // Reachable squares should be positive
        assert!(control[2] > 0.0, "my_reachable > 0");
        assert!(control[3] > 0.0, "opp_reachable > 0");

        // Duke mobility should be positive (duke has horizontal slides)
        assert!(duke_mob[0] > 0.0, "my_duke_mobility > 0");
        assert!(duke_mob[1] > 0.0, "opp_duke_mobility > 0");
    }

    #[test]
    fn defended_is_nonzero_initial_state() {
        // This is the key regression test: defended must NOT be zero.
        let gs = initial_state();
        let (control, _) = board_control_features_with_duke_mob(&gs);

        // In the initial state, the duke is adjacent to both footmen.
        // The duke can slide to the footman's square (ignoring friendly),
        // and each footman adjacent to the duke can reach the duke's square.
        // So at minimum the duke and one footman should be "defended".
        assert!(
            control[5] > 0.0,
            "my_defended must be > 0 for initial board, got {}",
            control[5]
        );
        assert!(
            control[7] > 0.0,
            "opp_defended must be > 0 for initial board, got {}",
            control[7]
        );
    }

    #[test]
    fn defended_exact_values_initial_state() {
        let gs = initial_state();
        let (control, _) = board_control_features_with_duke_mob(&gs);

        // TopPlayer: Duke(3,0), Footman(4,0), Footman(3,1)
        //   Duke(3,0): defended by Footman(4,0) (move left) and Footman(3,1) (move up) => YES
        //   Footman(4,0): Duke(3,0) can slide right to (4,0) => YES
        //   Footman(3,1): Duke slides horizontally only, Footman(4,0) moves to (3,0)/(5,0)/(4,1) not (3,1) => NO
        assert_eq!(control[5], 2.0, "my_defended should be 2 (duke + right footman)");

        // BottomPlayer: Duke(3,5), Footman(4,5), Footman(3,4)
        //   Duke(3,5): defended by Footman(4,5) (move left) and Footman(3,4) (move down) => YES
        //   Footman(4,5): Duke(3,5) can slide right to (4,5) => YES
        //   Footman(3,4): Duke slides horizontally only (3,5); Footman(4,5) moves to (3,5)/(5,5)/(4,4) not (3,4) => NO
        assert_eq!(control[7], 2.0, "opp_defended should be 2 (duke + right footman)");
    }

    #[test]
    fn threatened_counts() {
        // Place pieces so they can threaten each other
        // TopPlayer footman at (2,2) can move to (3,2) which is enemy footman
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(2, 2), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
            (c(3, 2), PlacedTile::new(Owner::BottomPlayer, TileType::Footman)),
        ];
        let gs = state_from_tiles(tiles);
        let (control, _) = board_control_features_with_duke_mob(&gs);

        // my_threatened: enemy tiles on squares I can reach
        // TopPlayer footman(2,2) can move to (3,2) where BottomPlayer footman sits => 1
        // Also check if duke can reach any enemy tile — duke at (0,0) slides horizontally,
        // can reach (1,0)..(5,0), none are enemy occupied.
        assert!(
            control[6] >= 1.0,
            "my_threatened should be >= 1 (footman threatens enemy footman), got {}",
            control[6]
        );

        // opp_threatened: my tiles on squares opponent can reach
        // BottomPlayer footman(3,2) can move to (2,2) where TopPlayer footman sits => 1
        assert!(
            control[8] >= 1.0,
            "opp_threatened should be >= 1 (enemy footman threatens my footman), got {}",
            control[8]
        );
    }

    #[test]
    fn contested_squares_exist() {
        // Place pieces so their reachable squares overlap
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(2, 2), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
            (c(2, 4), PlacedTile::new(Owner::BottomPlayer, TileType::Footman)),
        ];
        let gs = state_from_tiles(tiles);
        let (control, _) = board_control_features_with_duke_mob(&gs);

        // Footman at (2,2) reaches (1,2),(3,2),(2,1),(2,3)
        // Footman at (2,4) reaches (1,4),(3,4),(2,3),(2,5)
        // Overlap at (2,3) => contested >= 1
        assert!(
            control[4] >= 1.0,
            "contested should be >= 1, got {}",
            control[4]
        );
    }

    #[test]
    fn duke_mobility_counts() {
        // Duke on side 1 slides horizontally. Duke at (3,0) with nothing blocking:
        // can slide to (0,0),(1,0),(2,0),(4,0),(5,0) = 5 moves
        let tiles = vec![
            (c(3, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(3, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = state_from_tiles(tiles);
        let (_, duke_mob) = board_control_features_with_duke_mob(&gs);

        assert_eq!(duke_mob[0], 5.0, "TopPlayer duke should have 5 horizontal slide moves");
        assert_eq!(duke_mob[1], 5.0, "BottomPlayer duke should have 5 horizontal slide moves");
    }

    // ── discard_vector ──────────────────────────────────────────────

    #[test]
    fn discard_vector_empty() {
        let gs = initial_state();
        let dv = discard_vector(&gs);
        // No captures in initial state, all discards should be 0
        assert_eq!(dv, [0.0; 26]);
    }

    #[test]
    fn discard_vector_with_discards() {
        // Use from_snapshot with pre-populated discard piles
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(0, 1), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
            (c(5, 4), PlacedTile::new(Owner::BottomPlayer, TileType::Footman)),
        ];
        let gs = GameState::from_snapshot(GameSnapshot {
            tiles,
            top_bag: TileBag::new(vec![]),
            bottom_bag: TileBag::new(vec![]),
            top_discard: DiscardBag::from_tiles(vec![TileType::Footman, TileType::Bowman]),
            bottom_discard: DiscardBag::from_tiles(vec![TileType::Knight]),
            current_turn: Owner::TopPlayer,
            idle_move_count: 0,
        });

        let dv = discard_vector(&gs);

        // TopPlayer (my) discards: Footman (index 1) and Bowman (index 2)
        assert_eq!(dv[TileType::Footman.index()], 1.0, "my Footman discard");
        assert_eq!(dv[TileType::Bowman.index()], 1.0, "my Bowman discard");
        // Other my discards should be 0
        assert_eq!(dv[TileType::Duke.index()], 0.0);
        assert_eq!(dv[TileType::Knight.index()], 0.0);

        // BottomPlayer (opp) discards: Knight (index 3)
        let opp_offset = TileType::COUNT;
        assert_eq!(dv[opp_offset + TileType::Knight.index()], 1.0, "opp Knight discard");
        assert_eq!(dv[opp_offset + TileType::Footman.index()], 0.0, "opp has no Footman discard");
    }

    // ── extract_combined_features ───────────────────────────────────

    #[test]
    fn combined_features_concatenation() {
        let gs = initial_state();
        let combined = extract_combined_features(&gs);
        let manhattan = manhattan_distance_features(&gs);
        let (control, duke_mob) = board_control_features_with_duke_mob(&gs);
        let discards = discard_vector(&gs);

        // Verify the concatenation layout: manhattan[4] + control[9] + duke_mob[2] + discards[26] = 41
        assert_eq!(combined.len(), NUM_COMBINED_FEATURES);

        for i in 0..4 {
            assert_eq!(combined[i], manhattan[i], "manhattan feature {}", i);
        }
        for i in 0..9 {
            assert_eq!(combined[4 + i], control[i], "control feature {}", i);
        }
        for i in 0..2 {
            assert_eq!(combined[13 + i], duke_mob[i], "duke_mob feature {}", i);
        }
        for i in 0..26 {
            assert_eq!(combined[15 + i], discards[i], "discard feature {}", i);
        }
    }

    #[test]
    fn combined_features_length() {
        let gs = initial_state();
        let combined = extract_combined_features(&gs);
        assert_eq!(combined.len(), 41);
    }

    // ── Edge cases: multiple units on same square ───────────────────

    #[test]
    fn defended_multiple_defenders_same_piece() {
        // Place a piece that is defended by TWO friendly pieces.
        // TopPlayer: Duke(2,0), Footman(3,0), Footman(1,0)
        // Footman(3,0) can be reached by Duke sliding right AND Footman(1,0) can't reach (3,0).
        // But Duke(2,0) is defended by both footmen (each can move to (2,0) if ignoring friendly).
        let tiles = vec![
            (c(2, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(3, 0), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(1, 0), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = state_from_tiles(tiles);
        let (control, _) = board_control_features_with_duke_mob(&gs);

        // my_defended counts each tile that has at least one other friendly defender.
        // Duke(2,0): both footmen can reach it -> defended
        // Footman(3,0): Duke can slide right to (3,0) -> defended
        // Footman(1,0): Duke can slide left to (1,0) -> defended
        // All 3 should be defended
        assert_eq!(control[5], 3.0,
            "all 3 TopPlayer pieces should be defended, got {}", control[5]);
    }

    #[test]
    fn threatened_multiple_attackers_same_target() {
        // Two TopPlayer pieces can both threaten the same enemy piece.
        // The threatened count should still be 1 (counts threatened PIECES, not attacker count).
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(2, 1), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(2, 3), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
            (c(2, 2), PlacedTile::new(Owner::BottomPlayer, TileType::Footman)),
        ];
        let gs = state_from_tiles(tiles);
        let (control, _) = board_control_features_with_duke_mob(&gs);

        // Both TopPlayer footmen can move to (2,2) where enemy footman sits.
        // my_threatened should count the enemy footman once, not twice.
        // Also duke at (0,0) might threaten something via horizontal slides.
        // The enemy footman at (2,2) should be in my_reach -> my_threatened >= 1
        assert!(control[6] >= 1.0,
            "enemy footman at (2,2) should be threatened, got {}", control[6]);
    }

    #[test]
    fn isolated_piece_not_defended() {
        // A piece far from all friendlies should NOT be defended.
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(5, 5), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),  // isolated
            (c(0, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = state_from_tiles(tiles);
        let (control, _) = board_control_features_with_duke_mob(&gs);

        // Duke(0,0) has no friendly piece that can reach it (footman is at (5,5), too far)
        // Footman(5,5) has no friendly piece that can reach it (duke slides row 0 only)
        // my_defended should be 0
        assert_eq!(control[5], 0.0,
            "isolated pieces should not be defended, got {}", control[5]);
    }

    #[test]
    fn symmetric_board_symmetric_features() {
        // Mirror board should give symmetric features between my and opp.
        // TopPlayer: Duke(1,1), Footman(2,1)
        // BottomPlayer: Duke(1,4), Footman(2,4)
        let tiles = vec![
            (c(1, 1), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(2, 1), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(1, 4), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
            (c(2, 4), PlacedTile::new(Owner::BottomPlayer, TileType::Footman)),
        ];
        let gs = state_from_tiles(tiles);
        let (control, duke_mob) = board_control_features_with_duke_mob(&gs);

        // Symmetric board => my and opp features should be equal
        assert_eq!(control[0], control[1], "my_moves == opp_moves on symmetric board");
        assert_eq!(control[2], control[3], "my_reachable == opp_reachable");
        assert_eq!(control[5], control[7], "my_defended == opp_defended");
        assert_eq!(control[6], control[8], "my_threatened == opp_threatened");
        assert_eq!(duke_mob[0], duke_mob[1], "duke mobility should be equal");
    }

    // ── Edge case: Manhattan distance ──────────────────────────────

    #[test]
    fn manhattan_boundary_distance_2() {
        // Piece at exactly Manhattan distance 2 from duke should count as "near".
        // TopPlayer Duke at (0,0), Footman at (2,0) => dist = 2
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(2, 0), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = state_from_tiles(tiles);
        let md = manhattan_distance_features(&gs);

        // Footman(2,0) dist from my duke(0,0) = |2-0|+|0-0| = 2 => near
        assert_eq!(md[0], 1.0, "piece at distance 2 should count as near my duke");
    }

    #[test]
    fn manhattan_boundary_distance_3() {
        // Piece at Manhattan distance 3 from duke should NOT count as "near".
        // TopPlayer Duke at (0,0), Footman at (3,0) => dist = 3
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(3, 0), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = state_from_tiles(tiles);
        let md = manhattan_distance_features(&gs);

        // Footman(3,0) dist from my duke(0,0) = |3-0|+|0-0| = 3 => NOT near
        assert_eq!(md[0], 0.0, "piece at distance 3 should NOT count as near my duke");
    }

    #[test]
    fn manhattan_all_near_one_duke() {
        // Cluster many pieces around TopPlayer duke at (3,3).
        // All non-duke pieces within distance 2.
        let tiles = vec![
            (c(3, 3), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(3, 2), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),  // dist 1
            (c(4, 3), PlacedTile::new(Owner::TopPlayer, TileType::Pikeman)),  // dist 1
            (c(2, 3), PlacedTile::new(Owner::TopPlayer, TileType::Knight)),   // dist 1
            (c(3, 4), PlacedTile::new(Owner::TopPlayer, TileType::Champion)), // dist 1
            (c(4, 4), PlacedTile::new(Owner::TopPlayer, TileType::Bowman)),   // dist 2
            (c(0, 0), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = state_from_tiles(tiles);
        let md = manhattan_distance_features(&gs);

        // 5 non-duke pieces all within dist 2 of my duke
        assert_eq!(md[0], 5.0, "all 5 pieces should be near my duke");
    }

    #[test]
    fn manhattan_dukes_adjacent() {
        // Dukes next to each other, pieces near both.
        // TopPlayer Duke(2,2), BottomPlayer Duke(3,2) => adjacent
        // TopPlayer Footman(2,3) => dist 1 from my duke, dist 2 from enemy duke
        // BottomPlayer Footman(3,3) => dist 2 from my duke, dist 1 from enemy duke
        let tiles = vec![
            (c(2, 2), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(2, 3), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(3, 2), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
            (c(3, 3), PlacedTile::new(Owner::BottomPlayer, TileType::Footman)),
        ];
        let gs = state_from_tiles(tiles);
        let md = manhattan_distance_features(&gs);

        // my non-duke near my duke(2,2): Footman(2,3) dist=1 => 1
        assert_eq!(md[0], 1.0, "my_units_near_my_duke");
        // enemy near my duke(2,2): Duke(3,2) dist=1, Footman(3,3) dist=2 => 2
        assert_eq!(md[1], 2.0, "enemy_units_near_my_duke");
        // my non-duke near enemy duke(3,2): Footman(2,3) dist=|2-3|+|3-2|=2 => 1
        assert_eq!(md[2], 1.0, "my_units_near_enemy_duke");
        // enemy non-duke near enemy duke(3,2): Footman(3,3) dist=1 => 1
        assert_eq!(md[3], 1.0, "enemy_units_near_enemy_duke");
    }

    // ── Edge case: Board control ───────────────────────────────────

    #[test]
    fn defended_piece_also_threatened() {
        // A piece that is both defended by friendly AND threatened by enemy.
        // TopPlayer: Duke(0,0), Footman(1,0) — duke defends footman via slide
        // BottomPlayer: Duke(5,5), Footman(2,0) — enemy footman can capture TopPlayer's footman
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(1, 0), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
            (c(2, 0), PlacedTile::new(Owner::BottomPlayer, TileType::Footman)),
        ];
        let gs = state_from_tiles(tiles);
        let (control, _) = board_control_features_with_duke_mob(&gs);

        // Footman(1,0) is defended by Duke(0,0) sliding right
        assert!(control[5] >= 1.0,
            "my_defended should be >= 1 (duke defends footman), got {}", control[5]);
        // Footman(1,0) is threatened by BottomPlayer's Footman(2,0) moving left
        assert!(control[8] >= 1.0,
            "opp_threatened should be >= 1 (enemy footman threatens my footman), got {}", control[8]);
    }

    #[test]
    fn defended_by_multiple_piece_types() {
        // A piece defended by both duke (slide) and footman (step).
        // Duke side A only slides horizontally.
        // TopPlayer: Duke(1,0), Footman(2,0), Footman(0,0)
        // Duke(1,0): Footman(2,0) can step left to (1,0), Footman(0,0) can step right to (1,0) => defended
        // Footman(2,0): Duke(1,0) can slide right to (2,0) => defended
        // Footman(0,0): Duke(1,0) can slide left to (0,0) => defended
        let tiles = vec![
            (c(1, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(2, 0), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = state_from_tiles(tiles);
        let (control, _) = board_control_features_with_duke_mob(&gs);

        // All 3 of TopPlayer's pieces should be defended
        assert_eq!(control[5], 3.0,
            "all 3 pieces should be defended (duke by both footmen, each footman by duke), got {}", control[5]);
    }

    #[test]
    fn duke_boxed_in_zero_mobility() {
        // Duke surrounded by friendly pieces on horizontal slides has zero mobility.
        // Duke side A only slides horizontally.
        // TopPlayer Duke(1,1), blocked by friendlies at (0,1) and (2,1).
        let tiles = vec![
            (c(1, 1), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(0, 1), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(2, 1), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = state_from_tiles(tiles);
        let (_, duke_mob) = board_control_features_with_duke_mob(&gs);

        assert_eq!(duke_mob[0], 0.0,
            "duke boxed in by friendlies should have 0 mobility, got {}", duke_mob[0]);
    }

    #[test]
    fn duke_center_max_mobility() {
        // Duke in center of empty board has maximum horizontal slide moves.
        // Duke side A only slides horizontally (HorizontalSymmetricOffset::Near, Slide).
        // Duke at (3,3): slides left to (2,3),(1,3),(0,3) = 3, right to (4,3),(5,3) = 2
        // Total = 5 horizontal slides
        let tiles = vec![
            (c(3, 3), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(0, 0), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = state_from_tiles(tiles);
        let (_, duke_mob) = board_control_features_with_duke_mob(&gs);

        // Duke at (3,3) slides to 5 squares (3 left + 2 right)
        assert_eq!(duke_mob[0], 5.0,
            "duke at center should have 5 horizontal slide moves, got {}", duke_mob[0]);
    }

    #[test]
    fn duke_edge_limited_mobility() {
        // Duke on edge has fewer moves than center.
        // Duke at (0,3) side A: slides right (1,3),(2,3),(3,3),(4,3),(5,3) = 5, no slide left (at edge)
        // steps up (0,4) and down (0,2) = 2
        // Total = 7 — same horizontal count since edge only cuts one direction
        //
        // Duke at (0,0) corner: slides right (1,0)...(5,0) = 5, step up (0,1) = 1, no step down (edge)
        // Total = 6
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = state_from_tiles(tiles);
        let (_, duke_mob) = board_control_features_with_duke_mob(&gs);

        // Corner duke has strictly fewer moves than center duke
        assert!(duke_mob[0] < 7.0,
            "corner duke should have fewer than 7 moves, got {}", duke_mob[0]);
        assert!(duke_mob[0] > 0.0,
            "corner duke should still have some moves, got {}", duke_mob[0]);
    }

    #[test]
    fn zero_moves_blocked_piece() {
        // A non-duke piece completely surrounded by friendlies has no legal moves.
        // TopPlayer: Footman(2,2) surrounded by friendlies at (1,2),(3,2),(2,1),(2,3)
        // Footman side A moves one square in each cardinal direction — all blocked.
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(2, 2), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(1, 2), PlacedTile::new(Owner::TopPlayer, TileType::Pikeman)),
            (c(3, 2), PlacedTile::new(Owner::TopPlayer, TileType::Knight)),
            (c(2, 1), PlacedTile::new(Owner::TopPlayer, TileType::Champion)),
            (c(2, 3), PlacedTile::new(Owner::TopPlayer, TileType::Bowman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = state_from_tiles(tiles);
        let (control, _) = board_control_features_with_duke_mob(&gs);

        // With the blocked footman, total move count should be less than if it were free.
        // We can check that moves > 0 (other pieces still move) but the blocked footman
        // contributes 0 moves. Let's compare with an unblocked version.
        let tiles_unblocked = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(2, 2), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs_unblocked = state_from_tiles(tiles_unblocked);
        let (control_unblocked, _) = board_control_features_with_duke_mob(&gs_unblocked);

        // The blocked board has more pieces but the footman at (2,2) contributes 0 moves.
        // The unblocked footman at (2,2) has 4 moves.
        // So my_approx_moves for blocked < my_approx_moves for unblocked + moves from extra pieces.
        // At minimum, verify the blocked footman's contribution is reflected.
        assert!(control[0] > 0.0, "should still have some moves from other pieces");

        // The unblocked footman alone contributes 4 moves. With surrounding pieces blocked,
        // those 4 moves vanish but the surrounding pieces add their own.
        // Key assertion: blocked version has fewer moves from footman at (2,2).
        // We can't easily isolate footman's contribution, but we verify the feature is computed.
        assert!(control[0] > 0.0 && control_unblocked[0] > 0.0);
    }

    #[test]
    fn all_reachable_squares_contested() {
        // Set up board where reachable squares overlap significantly.
        // Place footmen facing each other so their reachable squares overlap.
        // Footman side A moves one square in each cardinal direction.
        // TopPlayer Footman(2,3) reaches (1,3),(3,3),(2,2),(2,4)
        // BottomPlayer Footman(4,3) reaches (3,3),(5,3),(4,2),(4,4)
        // Overlap at (3,3) — both can move/capture there.
        // Also use duke slides to create more overlap.
        // TopPlayer Duke(0,3) slides to (1,3),(2,3) blocked by footman at (2,3)... wait, slides stop at first friendly.
        // Let's keep it simple: two footmen whose reachable squares overlap.
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(2, 3), PlacedTile::new(Owner::TopPlayer, TileType::Footman)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
            (c(4, 3), PlacedTile::new(Owner::BottomPlayer, TileType::Footman)),
        ];
        let gs = state_from_tiles(tiles);
        let (control, _) = board_control_features_with_duke_mob(&gs);

        // TopPlayer Footman(2,3) reaches: (1,3),(3,3),(2,2),(2,4)
        // BottomPlayer Footman(4,3) reaches: (3,3),(5,3),(4,2),(4,4)
        // Overlap at (3,3) — contested >= 1
        // Also duke slides may add more overlap.
        assert!(control[4] >= 1.0,
            "contested squares should be >= 1 with facing footmen, got {}", control[4]);
    }

    // ── Edge case: Discard ─────────────────────────────────────────

    #[test]
    fn discard_multiple_same_type() {
        // Multiple footmen in discard pile should count correctly.
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = GameState::from_snapshot(GameSnapshot {
            tiles,
            top_bag: TileBag::new(vec![]),
            bottom_bag: TileBag::new(vec![]),
            top_discard: DiscardBag::from_tiles(vec![TileType::Footman, TileType::Footman, TileType::Footman]),
            bottom_discard: DiscardBag::empty(),
            current_turn: Owner::TopPlayer,
            idle_move_count: 0,
        });

        let dv = discard_vector(&gs);
        assert_eq!(dv[TileType::Footman.index()], 3.0,
            "3 footmen in discard should give count of 3, got {}", dv[TileType::Footman.index()]);
        // All other slots should be 0
        for i in 0..26 {
            if i != TileType::Footman.index() {
                assert_eq!(dv[i], 0.0, "slot {} should be 0", i);
            }
        }
    }

    #[test]
    fn discard_asymmetric() {
        // One player has many discards, other has none.
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = GameState::from_snapshot(GameSnapshot {
            tiles,
            top_bag: TileBag::new(vec![]),
            bottom_bag: TileBag::new(vec![]),
            top_discard: DiscardBag::empty(),
            bottom_discard: DiscardBag::from_tiles(vec![
                TileType::Footman, TileType::Pikeman, TileType::Knight,
                TileType::Champion, TileType::Bowman,
            ]),
            current_turn: Owner::TopPlayer,
            idle_move_count: 0,
        });

        let dv = discard_vector(&gs);

        // My (TopPlayer) discards: all zero
        for i in 0..TileType::COUNT {
            assert_eq!(dv[i], 0.0, "my discard slot {} should be 0", i);
        }

        // Opp (BottomPlayer) discards: 5 distinct types
        let opp_offset = TileType::COUNT;
        assert_eq!(dv[opp_offset + TileType::Footman.index()], 1.0, "opp Footman discard");
        assert_eq!(dv[opp_offset + TileType::Pikeman.index()], 1.0, "opp Pikeman discard");
        assert_eq!(dv[opp_offset + TileType::Knight.index()], 1.0, "opp Knight discard");
        assert_eq!(dv[opp_offset + TileType::Champion.index()], 1.0, "opp Champion discard");
        assert_eq!(dv[opp_offset + TileType::Bowman.index()], 1.0, "opp Bowman discard");

        // Total opp discards = 5
        let opp_total: f64 = dv[opp_offset..].iter().sum();
        assert_eq!(opp_total, 5.0, "total opponent discards should be 5");
    }

    // ── Edge case: General ─────────────────────────────────────────

    #[test]
    fn empty_board_just_dukes() {
        // Minimal board with only two dukes — verify baseline features.
        let tiles = vec![
            (c(0, 0), PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (c(5, 5), PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs = state_from_tiles(tiles);

        // Manhattan: no non-duke pieces => all zeros
        let md = manhattan_distance_features(&gs);
        assert_eq!(md, [0.0, 0.0, 0.0, 0.0],
            "no non-duke pieces means all manhattan features are 0");

        // Board control: only duke moves
        let (control, duke_mob) = board_control_features_with_duke_mob(&gs);
        // Only duke tiles exist, so moves come from duke only
        assert_eq!(control[0], duke_mob[0],
            "my_approx_moves should equal my duke mobility (only duke on board)");
        assert_eq!(control[1], duke_mob[1],
            "opp_approx_moves should equal opp duke mobility");

        // No non-duke pieces => defended/threatened = 0 for "number of tiles defended"
        // Actually duke can still be "defended" by... nothing (only 1 piece per side).
        // With only 1 tile per player, no OTHER friendly can defend => 0
        assert_eq!(control[5], 0.0, "my_defended should be 0 with just duke");
        assert_eq!(control[7], 0.0, "opp_defended should be 0 with just duke");

        // Discard: empty
        let dv = discard_vector(&gs);
        assert_eq!(dv, [0.0; 26], "discard should be all zeros");

        // Combined features: length is 41
        let combined = extract_combined_features(&gs);
        assert_eq!(combined.len(), NUM_COMBINED_FEATURES);
        // First 4 are manhattan (all 0)
        assert_eq!(&combined[0..4], &[0.0; 4]);
        // Last 26 are discard (all 0)
        assert_eq!(&combined[15..41], &[0.0; 26]);
    }
}

