//! Learned heuristic evaluator using polynomial feature expansion + ridge regression.
//!
//! 24 features total:
//! - 4 base heuristic differences (approx, for speed)
//! - 14 polynomial expansion terms (quadratic + cross + cubic)
//! - 5 new cheap features
//! - 1 bias term

use std::fs;

use duke_rust::game::ai::heuristics::{Heuristic, Heuristics};
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

    // x7: tile_adjacency_diff
    let my_adj = count_adjacent_pairs(gs, owner) as f64;
    let opp_adj = count_adjacent_pairs(gs, opp) as f64;
    let x7 = my_adj - opp_adj;

    // x8: center_control_diff
    let my_center = count_center_tiles(gs, owner) as f64;
    let opp_center = count_center_tiles(gs, opp) as f64;
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

/// Count orthogonally adjacent own-tile pairs for a given owner.
fn count_adjacent_pairs(gs: &GameState, owner: Owner) -> usize {
    let tiles = gs.get_tiles_for_owner(owner);
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

/// Count tiles in the center 4 squares for a given owner.
fn count_center_tiles(gs: &GameState, owner: Owner) -> usize {
    let tiles = gs.get_tiles_for_owner(owner);
    tiles.iter()
        .filter(|(c, _)| CENTER_SQUARES.contains(&(c.x, c.y)))
        .count()
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

    /// Add one game trajectory to the accumulator.
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
            for i in 0..NUM_FEATURES {
                self.xty[i] += features[i] * target;
                // Upper triangle only (X'X is symmetric)
                for j in i..NUM_FEATURES {
                    self.xtx[i][j] += features[i] * features[j];
                }
            }
            self.n_samples += 1;
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
