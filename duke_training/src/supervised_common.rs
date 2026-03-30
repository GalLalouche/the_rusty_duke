//! Shared utilities for supervised training binaries.
//!
//! Contains data structures and functions used by multiple supervised trainers
//! (FC, CNN, burn-CNN), extracted to avoid code duplication.

use std::time::Instant;

use crate::encoding::BAG_FEATURES;
use crate::game_setup::{create_bag, create_initial_state};
use crate::loaded_model::LoadedModel;
use crate::match_runner::{run_matches, win_rate, Player};

// ── Labeled position data ────────────────────────────────────────────────

/// A single labeled position loaded from an LPOS or FLPS binary file.
///
/// Each position stores sparse board feature indices, dense bag features,
/// a scalar label (e.g. minimax evaluation), and a frequency count.
pub struct LabeledPosition {
    /// Active board feature indices (each < 1080).
    pub active_indices: Vec<u16>,
    /// Dense bag features (26 f32 values).
    pub bag_features: [f32; BAG_FEATURES],
    /// Evaluation label (e.g. LR-Cheap depth-2 minimax score).
    pub label: f32,
    /// Frequency weight (how many times this position appeared).
    pub count: u32,
}

/// Human-readable names for the 41 combined features, used when loading FLPS files
/// to display which label index was selected.
pub const COMBINED_FEATURE_NAMES: [&str; 41] = [
    "near_my_duke_friendly", "near_my_duke_enemy",
    "near_enemy_duke_friendly", "near_enemy_duke_enemy",
    "my_moves", "opp_moves", "my_reachable", "opp_reachable", "contested",
    "my_defended", "my_threatened", "opp_defended", "opp_threatened",
    "my_duke_mob", "opp_duke_mob",
    "my_duke_disc", "my_footman_disc", "my_pikeman_disc", "my_knight_disc",
    "my_sergeant_disc", "my_ranger_disc", "my_champion_disc", "my_wizard_disc",
    "my_general_disc", "my_marshall_disc", "my_assassin_disc", "my_longbowman_disc",
    "my_dragoon_disc",
    "opp_duke_disc", "opp_footman_disc", "opp_pikeman_disc", "opp_knight_disc",
    "opp_sergeant_disc", "opp_ranger_disc", "opp_champion_disc", "opp_wizard_disc",
    "opp_general_disc", "opp_marshall_disc", "opp_assassin_disc", "opp_longbowman_disc",
    "opp_dragoon_disc",
];

/// Load labeled positions from an LPOS or FLPS binary file.
///
/// Supports two file formats:
/// - **LPOS**: single label per position (label_index is ignored)
/// - **FLPS**: multiple labels per position; `label_index` selects which one to use
///
/// Returns `(positions, min_label, max_label)`.
pub fn load_lpos(path: &str, label_index: usize) -> (Vec<LabeledPosition>, f32, f32) {
    let t0 = Instant::now();
    eprintln!("Loading labeled positions from {} ...", path);

    let data = std::fs::read(path).expect("Failed to read labeled positions file");
    let mut cursor = 0usize;

    macro_rules! read_bytes {
        ($n:expr) => {{
            let end = cursor + $n;
            assert!(end <= data.len(), "Unexpected EOF at offset {}", cursor);
            let slice = &data[cursor..end];
            cursor = end;
            slice
        }};
    }
    macro_rules! read_u16 {
        () => {{
            let b = read_bytes!(2);
            u16::from_le_bytes([b[0], b[1]])
        }};
    }
    macro_rules! read_u32 {
        () => {{
            let b = read_bytes!(4);
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }};
    }
    macro_rules! read_f32 {
        () => {{
            let b = read_bytes!(4);
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }};
    }

    // Detect format by magic bytes
    let magic = read_bytes!(4);
    let is_flps = magic == b"FLPS";
    let is_lpos = magic == b"LPOS";
    assert!(is_lpos || is_flps,
        "Unknown file format (magic: {:?}), expected LPOS or FLPS", magic);

    let version = read_u32!();
    assert_eq!(version, 1, "Unsupported version {}", version);
    let num_positions = read_u32!() as usize;

    let num_labels = if is_flps {
        let nl = read_u32!() as usize;
        assert!(label_index < nl,
            "--label-index {} out of range (file has {} labels)", label_index, nl);
        let label_name = if nl == 41 && label_index < COMBINED_FEATURE_NAMES.len() {
            COMBINED_FEATURE_NAMES[label_index]
        } else {
            "unknown"
        };
        eprintln!("  FLPS format: {} positions, {} labels, using label index {} ({})",
            num_positions, nl, label_index, label_name);
        nl
    } else {
        if label_index != 0 {
            eprintln!("  Warning: --label-index {} ignored for LPOS format (single label)", label_index);
        }
        eprintln!("  LPOS format: {} positions, version {}", num_positions, version);
        1
    };

    let mut positions = Vec::with_capacity(num_positions);
    for _ in 0..num_positions {
        let num_active = read_u16!() as usize;
        let mut active_indices = Vec::with_capacity(num_active);
        for _ in 0..num_active {
            active_indices.push(read_u16!());
        }
        let mut bag_features = [0.0f32; BAG_FEATURES];
        for i in 0..BAG_FEATURES {
            bag_features[i] = read_f32!();
        }

        let label = if is_flps {
            // Read all labels, pick the one at label_index
            let mut selected = 0.0f32;
            for li in 0..num_labels {
                let val = read_f32!();
                if li == label_index {
                    selected = val;
                }
            }
            selected
        } else {
            read_f32!()
        };

        let count = read_u32!();
        positions.push(LabeledPosition {
            active_indices,
            bag_features,
            label,
            count,
        });
    }

    let elapsed = t0.elapsed();
    let file_mb = data.len() as f64 / (1024.0 * 1024.0);
    eprintln!(
        "  Loaded {} positions ({:.1} MB) in {:.1}s",
        positions.len(), file_mb, elapsed.as_secs_f64()
    );

    // Single-pass label stats computation instead of 3 separate iterations.
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut total_count = 0u64;
    for p in &positions {
        let l = p.label;
        if l < min { min = l; }
        if l > max { max = l; }
        sum += l as f64;
        total_count += p.count as u64;
    }
    let mean = sum / positions.len() as f64;
    eprintln!(
        "  Label stats: min={:.2}, max={:.2}, mean={:.4}, total_count={}",
        min, max, mean, total_count
    );

    (positions, min, max)
}

// ── Label normalization ──────────────────────────────────────────────────

/// Clamp threshold for label-to-target mapping.
/// Labels are clamped to `[-LABEL_CLAMP, +LABEL_CLAMP]` before linear mapping to `[0, 1]`.
pub const LABEL_CLAMP: f32 = 10.0;

/// Map a raw evaluation label to a training target in `[0, 1]`.
///
/// Clamps the label to `[-LABEL_CLAMP, +LABEL_CLAMP]`, then maps linearly to `[0, 1]`.
/// Normal positions (roughly +/-6) get good spread (0.2..0.8).
/// Terminal positions (e.g. +/-30) clamp to 0 or 1.
#[inline]
pub fn label_to_target(label: f32) -> f32 {
    let clamped = label.clamp(-LABEL_CLAMP, LABEL_CLAMP);
    (clamped + LABEL_CLAMP) / (2.0 * LABEL_CLAMP)
}

// ── Adam optimizer ───────────────────────────────────────────────────────

/// Adam optimizer state for gradient-based training.
///
/// Uses running products of `beta1^t` and `beta2^t` to avoid calling `powf` each step.
pub struct AdamState {
    /// First moment estimates.
    pub m: Vec<f32>,
    /// Second moment estimates.
    pub v: Vec<f32>,
    /// Time step counter.
    pub t: u64,
    /// Learning rate (may be adjusted externally for adaptive LR schedules).
    pub lr: f32,
    /// First moment decay rate (default 0.9).
    pub beta1: f32,
    /// Second moment decay rate (default 0.999).
    pub beta2: f32,
    /// Numerical stability constant (default 1e-8).
    pub eps: f32,
    /// Running product of beta1^t (avoids powf per step).
    beta1_t: f32,
    /// Running product of beta2^t (avoids powf per step).
    beta2_t: f32,
}

impl AdamState {
    /// Create a new Adam optimizer for `num_params` parameters.
    pub fn new(num_params: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            beta1_t: 1.0,
            beta2_t: 1.0,
        }
    }

    /// Perform one Adam update step.
    ///
    /// `grad` should be the mean gradient over the batch (same length as `weights`).
    /// Updates weights in-place using bias-corrected first and second moment estimates.
    pub fn step(&mut self, weights: &mut [f32], grad: &[f32]) {
        self.t += 1;
        self.beta1_t *= self.beta1;
        self.beta2_t *= self.beta2;
        let lr_t = self.lr * (1.0 - self.beta2_t).sqrt() / (1.0 - self.beta1_t);
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let one_minus_beta1 = 1.0 - beta1;
        let one_minus_beta2 = 1.0 - beta2;
        let eps = self.eps;
        let n = weights.len();

        // Split into 3 passes for better cache utilization on large param vectors
        for i in 0..n {
            self.m[i] = beta1 * self.m[i] + one_minus_beta1 * grad[i];
        }
        for i in 0..n {
            self.v[i] = beta2 * self.v[i] + one_minus_beta2 * grad[i] * grad[i];
        }
        for i in 0..n {
            weights[i] -= lr_t * self.m[i] / (self.v[i].sqrt() + eps);
        }
    }
}

// ── Weighted index builder ───────────────────────────────────────────────

/// Build a weighted index array for sampling positions during training.
///
/// Each position index is repeated according to its `count`, so more frequent
/// positions are sampled proportionally more often. If the total weighted count
/// exceeds 200M, falls back to sqrt-weighted sampling to limit memory usage.
///
/// Returns `(indices, effective_total)` where `indices[i]` is a position index
/// into the positions array, and `effective_total` is the length of the array.
pub fn build_weighted_indices(positions: &[LabeledPosition]) -> (Vec<u32>, usize) {
    let total_weighted: usize = positions.iter().map(|p| p.count as usize).sum();

    if total_weighted <= 200_000_000 {
        let mut indices: Vec<u32> = Vec::with_capacity(total_weighted);
        for (i, pos) in positions.iter().enumerate() {
            for _ in 0..pos.count {
                indices.push(i as u32);
            }
        }
        let len = indices.len();
        (indices, len)
    } else {
        eprintln!(
            "  Total weighted count {} exceeds 200M, using sqrt-weighted sampling",
            total_weighted
        );
        let mut indices: Vec<u32> = Vec::new();
        for (i, pos) in positions.iter().enumerate() {
            let repeats = (pos.count as f64).sqrt().ceil() as u32;
            for _ in 0..repeats {
                indices.push(i as u32);
            }
        }
        let len = indices.len();
        (indices, len)
    }
}

// ── Benchmark evaluation ─────────────────────────────────────────────────

/// Run benchmark matches between a model player and a set of opponent specs.
///
/// Plays `eval_games` matches against each opponent specified in `benchmark_specs`
/// and prints win rates to stderr. The `model_player` should be a `Player::Evaluator`
/// wrapping the model under test.
pub fn run_benchmark(
    model_player: &Player<'_>,
    benchmark_specs: &[String],
    eval_games: u32,
) {
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    for spec in benchmark_specs {
        let opponent = LoadedModel::from_spec(spec, false);
        let opp_player = opponent.as_player();
        let label = &opponent.label;
        let result = run_matches(
            &gs,
            model_player,
            &opp_player,
            eval_games,
            &format!("vs {}", label),
        );
        let wr = win_rate(result.player_a_wins, result.ties, eval_games);
        eprintln!(
            "  vs {}: {:.1}% win rate ({} W / {} L / {} T)",
            label,
            wr * 100.0,
            result.player_a_wins,
            result.player_b_wins,
            result.ties,
        );
    }
}
