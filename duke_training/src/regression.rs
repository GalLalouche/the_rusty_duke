//! Regression and linear algebra utilities for ridge regression training.
//!
//! Contains `RegressionAccumulator` for incremental X'X / X'y accumulation,
//! `solve_linear_system` for Gaussian elimination, and `train_weights` for
//! end-to-end ridge regression from game trajectories.

use duke_rust::game::state::{GameResult, GameState};

use crate::game_setup::game_result_target;
use crate::learned_heuristic::{
    extract_features, LearnedHeuristicWeights, NUM_FEATURES,
};

/// Ridge regularization parameter for numerical stability.
const LAMBDA: f64 = 1e-6;

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
            let target = game_result_target(*result, current).unwrap_or(0.0);
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
