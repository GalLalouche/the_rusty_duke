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
use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::tile::Owner;

use crate::game_setup::GameEvaluator;

/// Total number of features in the full polynomial expansion.
pub const NUM_FEATURES: usize = 24;

/// Number of base features (4 heuristics + bias).
pub const NUM_BASE_FEATURES: usize = 5;

/// Ridge regularization parameter for numerical stability.
const LAMBDA: f64 = 1e-6;

/// Center squares on the 6x6 board: (2,2), (3,2), (2,3), (3,3).
const CENTER_SQUARES: [(u16, u16); 4] = [(2, 2), (3, 2), (2, 3), (3, 3)];

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

/// Extract just the 4 base heuristic differences + bias (5 features).
pub fn extract_base_features(gs: &GameState) -> [f64; NUM_BASE_FEATURES] {
    let owner = gs.current_player_turn();
    [
        Heuristics::DukeMovementOptions.approx_difference(owner, gs),
        Heuristics::TotalTilesOnBoard.approx_difference(owner, gs),
        Heuristics::TotalMovementOptions.approx_difference(owner, gs),
        Heuristics::DiscardedUnits.approx_difference(owner, gs),
        1.0, // bias
    ]
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
        let is_duke_tile = tile.tile.tile_type().is_duke();

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

/// Number of cheap features (no move generation, no guard checking).
pub const NUM_CHEAP_FEATURES: usize = 15;

/// Extract only cheap features — O(tiles), no move generation or guard checking.
///
/// Returns 15 features:
///  [0] my_tile_count
///  [1] opp_tile_count
///  [2] my_bag_size
///  [3] opp_bag_size
///  [4] my_discard_count
///  [5] opp_discard_count
///  [6] my_adjacency (orthogonal adjacent own-tile pairs)
///  [7] opp_adjacency
///  [8] my_center_control (tiles in center 4 squares)
///  [9] opp_center_control
/// [10] my_units_near_my_duke (Manhattan dist <= 2, excl duke)
/// [11] enemy_units_near_my_duke
/// [12] my_units_near_enemy_duke (excl duke)
/// [13] enemy_units_near_enemy_duke (excl duke)
/// [14] bias (always 1.0)
pub fn extract_cheap_features(gs: &GameState) -> [f64; NUM_CHEAP_FEATURES] {
    let me = gs.current_player_turn();
    let opp = me.next_player();

    let my_tiles = gs.get_tiles_for_owner(me);
    let opp_tiles = gs.get_tiles_for_owner(opp);

    let my_tile_count = my_tiles.len() as f64;
    let opp_tile_count = opp_tiles.len() as f64;

    let my_bag = gs.bag_for_owner(me).remaining().len() as f64;
    let opp_bag = gs.bag_for_owner(opp).remaining().len() as f64;

    let my_discard = gs.discard_bag_for(me).len() as f64;
    let opp_discard = gs.discard_bag_for(opp).len() as f64;

    let my_adj = count_adjacent_pairs_from(&my_tiles) as f64;
    let opp_adj = count_adjacent_pairs_from(&opp_tiles) as f64;

    let my_center = count_center_tiles_from(&my_tiles) as f64;
    let opp_center = count_center_tiles_from(&opp_tiles) as f64;

    let manhattan = manhattan_distance_features(gs);

    [
        my_tile_count, opp_tile_count,
        my_bag, opp_bag,
        my_discard, opp_discard,
        my_adj, opp_adj,
        my_center, opp_center,
        manhattan[0], manhattan[1], manhattan[2], manhattan[3],
        1.0, // bias
    ]
}

/// Compute approx move counts and board control features in one pass.
/// Returns: [my_approx_moves, opp_approx_moves, my_reachable_squares, opp_reachable_squares, contested_squares]
///
/// Moves are computed while ignoring the guard constraint (the expensive part),
/// making this a cheap approximation. Only tile-move destinations (not placements)
/// contribute to the reachable-squares arrays.
pub fn board_control_features(gs: &GameState) -> [f64; 5] {
    let owner = gs.current_player_turn();
    let opp = owner.next_player();

    let mut my_reach = [false; 36];
    let mut opp_reach = [false; 36];

    // Only count tile-movement moves (not placements) so that approx_moves
    // is consistent with reachable_squares — both measure on-board tile actions.
    let mut my_approx_moves = 0u32;
    for pm in gs.all_valid_game_moves_for_ignoring_guard(owner) {
        if let PossibleMove::ApplyNonCommandTileAction { dst, .. } = &pm {
            my_approx_moves += 1;
            let idx = dst.y as usize * 6 + dst.x as usize;
            my_reach[idx] = true;
        }
    }

    let mut opp_approx_moves = 0u32;
    for pm in gs.all_valid_game_moves_for_ignoring_guard(opp) {
        if let PossibleMove::ApplyNonCommandTileAction { dst, .. } = &pm {
            opp_approx_moves += 1;
            let idx = dst.y as usize * 6 + dst.x as usize;
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

    [
        my_approx_moves as f64,
        opp_approx_moves as f64,
        my_reachable as f64,
        opp_reachable as f64,
        contested as f64,
    ]
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

/// Incrementally accumulates X'X and X'y for ridge regression.
/// Feed game trajectories in batches, then call `solve()` at the end.
pub struct RegressionAccumulator {
    xtx: [[f64; NUM_FEATURES]; NUM_FEATURES],
    xty: [f64; NUM_FEATURES],
    n_samples: u64,
}

impl RegressionAccumulator {
    pub fn new() -> Self {
        Self {
            xtx: [[0.0; NUM_FEATURES]; NUM_FEATURES],
            xty: [0.0; NUM_FEATURES],
            n_samples: 0,
        }
    }

    /// Add a single pre-extracted feature vector + target to the accumulator.
    pub fn add_sample(&mut self, features: &[f64; NUM_FEATURES], target: f64) {
        for i in 0..NUM_FEATURES {
            self.xty[i] += features[i] * target;
            for j in i..NUM_FEATURES {
                self.xtx[i][j] += features[i] * features[j];
            }
        }
        self.n_samples += 1;
    }

    /// Add one game trajectory to the accumulator (extracts features from GameStates).
    pub fn add_game(&mut self, states: &[GameState], result: &GameResult) {
        for state in states {
            if state.game_result() != GameResult::Ongoing {
                continue;
            }
            let features = extract_features(state);
            let current = state.current_player_turn();
            let target = match result {
                GameResult::Won(winner) => {
                    if *winner == current { 1.0 } else { -1.0 }
                }
                GameResult::Tie | GameResult::Ongoing => 0.0,
            };
            self.add_sample(&features, target);
        }
    }

    pub fn n_samples(&self) -> u64 { self.n_samples }

    /// Save the accumulated X'X and X'y matrices to a binary file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"XREG")?;
        f.write_all(&self.n_samples.to_le_bytes())?;
        for row in &self.xtx {
            for &val in row {
                f.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.xty {
            f.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load a previously saved accumulator.
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"XREG" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid regression file"));
        }
        let mut buf8 = [0u8; 8];
        f.read_exact(&mut buf8)?;
        let n_samples = u64::from_le_bytes(buf8);
        let mut xtx = [[0.0f64; NUM_FEATURES]; NUM_FEATURES];
        for row in &mut xtx {
            for val in row {
                f.read_exact(&mut buf8)?;
                *val = f64::from_le_bytes(buf8);
            }
        }
        let mut xty = [0.0f64; NUM_FEATURES];
        for val in &mut xty {
            f.read_exact(&mut buf8)?;
            *val = f64::from_le_bytes(buf8);
        }
        Ok(Self { xtx, xty, n_samples })
    }

    /// Solve the accumulated system with the default ridge lambda.
    pub fn solve(&self) -> LearnedHeuristicWeights {
        self.solve_with_lambda(LAMBDA)
    }

    /// Solve with a specific lambda value. Lambda is scaled by the average
    /// diagonal of X'X so that lambda=1.0 means "regularization strength equal
    /// to average feature energy". Without this, lambda has no effect because
    /// X'X entries are in the billions with millions of samples.
    pub fn solve_with_lambda(&self, lambda: f64) -> LearnedHeuristicWeights {
        if self.n_samples == 0 {
            return LearnedHeuristicWeights::default();
        }
        let avg_diag = (0..NUM_FEATURES)
            .map(|i| self.xtx[i][i])
            .sum::<f64>() / NUM_FEATURES as f64;
        let scaled_lambda = lambda * avg_diag.max(1.0);

        let mut xtx = self.xtx;
        let mut xty = self.xty;
        // Mirror upper triangle to lower (we only accumulated upper)
        for i in 0..NUM_FEATURES {
            for j in 0..i {
                xtx[i][j] = xtx[j][i];
            }
            xtx[i][i] += scaled_lambda;
        }
        let weights = solve_linear_system(&mut xtx, &mut xty);
        LearnedHeuristicWeights { weights }
    }
}

/// Train weights via ridge regression: w = (X'X + lambdaI)^{-1} X'y
///
/// Each game provides a sequence of (state, outcome) pairs.
/// The target y for each state is:
///   +1.0 if the current player at that state eventually won
///   -1.0 if the current player at that state eventually lost
///    0.0 for a tie
pub fn train_weights(games: &[(Vec<GameState>, GameResult)]) -> LearnedHeuristicWeights {
    let mut acc = RegressionAccumulator::new();
    for (states, result) in games {
        acc.add_game(states, result);
    }
    acc.solve()
}

/// Solve Aw = b for w using Gaussian elimination with partial pivoting.
///
/// Modifies A and b in place. Returns the solution vector.
pub fn solve_linear_system(
    a: &mut [[f64; NUM_FEATURES]; NUM_FEATURES],
    b: &mut [f64; NUM_FEATURES],
) -> [f64; NUM_FEATURES] {
    let n = NUM_FEATURES;

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            let val = a[row][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }

        let pivot = a[col][col];
        if pivot.abs() < 1e-15 {
            // Singular or near-singular; skip this column
            continue;
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut w = [0.0f64; NUM_FEATURES];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * w[j];
        }
        if a[i][i].abs() > 1e-15 {
            w[i] = sum / a[i][i];
        }
    }

    w
}
