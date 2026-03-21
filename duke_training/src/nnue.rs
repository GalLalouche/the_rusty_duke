use duke_rust::game::state::GameState;
use crate::encoding::{active_feature_indices, TOTAL_FEATURES, BOARD_FEATURES, bag_features, BAG_FEATURES};

pub const NUM_FEATURES: usize = TOTAL_FEATURES; // 1106
pub const L1_SIZE: usize = 256;
pub const L2_SIZE: usize = 32;
/// Maximum active features: 12 tiles × 2 features each (type + side).
pub const MAX_ACTIVE_FEATURES: usize = 24;

/// Raw model weights for NNUE inference.
///
/// L1 weights are stored in **column-major** order for cache-friendly access:
/// `l1_weight[feat * L1_SIZE + i]` = weight from input `feat` to hidden unit `i`.
/// This means `add_feature(feat)` reads a contiguous 1KB block (256 × f32).
pub struct NnueWeights {
    /// Layer 1 weights [NUM_FEATURES × L1_SIZE], column-major.
    /// l1_weight[feat * L1_SIZE + i] = weight from input feat to hidden unit i.
    pub l1_weight: Vec<f32>,
    /// Layer 1 bias [L1_SIZE].
    pub l1_bias: Vec<f32>,
    /// Layer 2 weights [L2_SIZE × L1_SIZE], row-major.
    pub l2_weight: Vec<f32>,
    /// Layer 2 bias [L2_SIZE].
    pub l2_bias: Vec<f32>,
    /// Layer 3 weights [1 × L2_SIZE], row-major.
    pub l3_weight: Vec<f32>,
    /// Layer 3 bias [1].
    pub l3_bias: Vec<f32>,
}

impl NnueWeights {
    const MAGIC: &'static [u8; 4] = b"DUKE";
    const VERSION: u32 = 2; // Bumped: column-major L1 layout

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(Self::MAGIC)?;
        f.write_all(&Self::VERSION.to_le_bytes())?;
        for v in [
            &self.l1_weight,
            &self.l1_bias,
            &self.l2_weight,
            &self.l2_bias,
            &self.l3_weight,
            &self.l3_bias,
        ] {
            for &val in v.iter() {
                f.write_all(&val.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        assert_eq!(&magic, Self::MAGIC, "Invalid NNUE file magic");
        let mut version = [0u8; 4];
        f.read_exact(&mut version)?;
        assert_eq!(
            u32::from_le_bytes(version),
            Self::VERSION,
            "Unsupported NNUE version (expected {}, got {})",
            Self::VERSION,
            u32::from_le_bytes(version),
        );

        let read_vec = |f: &mut std::fs::File, n: usize| -> std::io::Result<Vec<f32>> {
            let mut buf = vec![0u8; n * 4];
            f.read_exact(&mut buf)?;
            Ok(buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };

        Ok(Self {
            l1_weight: read_vec(&mut f, NUM_FEATURES * L1_SIZE)?,
            l1_bias: read_vec(&mut f, L1_SIZE)?,
            l2_weight: read_vec(&mut f, L2_SIZE * L1_SIZE)?,
            l2_bias: read_vec(&mut f, L2_SIZE)?,
            l3_weight: read_vec(&mut f, L2_SIZE)?, // 1 * L2_SIZE
            l3_bias: read_vec(&mut f, 1)?,
        })
    }
}

/// Cached first-layer output (pre-ReLU). 256 floats = 1KB, fits in L1 cache.
#[derive(Clone)]
pub struct NnueAccumulator {
    /// First layer output before ReLU: W1 * input + b1
    pub hidden: [f32; L1_SIZE],
}

impl NnueAccumulator {
    /// Build accumulator from a list of active feature indices.
    /// Starts from bias, then adds weight columns for each active feature.
    pub fn from_features(weights: &NnueWeights, features: &[usize]) -> Self {
        let mut hidden = [0.0f32; L1_SIZE];
        hidden.copy_from_slice(&weights.l1_bias);
        for &feat in features {
            debug_assert!(feat < NUM_FEATURES);
            // Column-major: l1_weight[feat * L1_SIZE .. (feat+1) * L1_SIZE]
            // is a contiguous 1KB block — cache-friendly.
            let col = &weights.l1_weight[feat * L1_SIZE..(feat + 1) * L1_SIZE];
            for i in 0..L1_SIZE {
                hidden[i] += col[i];
            }
        }
        Self { hidden }
    }

    /// Add a feature (tile placed/moved to a square).
    #[inline]
    pub fn add_feature(&mut self, feat: usize, weights: &NnueWeights) {
        let col = &weights.l1_weight[feat * L1_SIZE..(feat + 1) * L1_SIZE];
        for i in 0..L1_SIZE {
            self.hidden[i] += col[i];
        }
    }

    /// Remove a feature (tile removed/moved from a square).
    #[inline]
    pub fn remove_feature(&mut self, feat: usize, weights: &NnueWeights) {
        let col = &weights.l1_weight[feat * L1_SIZE..(feat + 1) * L1_SIZE];
        for i in 0..L1_SIZE {
            self.hidden[i] -= col[i];
        }
    }
}

/// NNUE evaluator for fast position evaluation.
pub struct NnueEvaluator {
    weights: NnueWeights,
}

impl NnueEvaluator {
    pub fn new(weights: NnueWeights) -> Self {
        Self { weights }
    }

    pub fn evaluate_state(&self, gs: &GameState) -> f32 {
        // Board features (sparse binary)
        let board_features = active_feature_indices(gs);
        let mut acc = NnueAccumulator::from_features(&self.weights, &board_features);

        // Bag features (dense, at indices BOARD_FEATURES..TOTAL_FEATURES)
        let bag = bag_features(gs);
        for (i, &val) in bag.iter().enumerate() {
            if val != 0.0 {
                let feat = BOARD_FEATURES + i;
                // Dense: multiply weight column by the count value
                let col = &self.weights.l1_weight[feat * L1_SIZE..(feat + 1) * L1_SIZE];
                for j in 0..L1_SIZE {
                    acc.hidden[j] += col[j] * val;
                }
            }
        }

        self.evaluate_from_accumulator(&acc)
    }

    /// Evaluate from a pre-computed accumulator (layers 2+3 only).
    /// This is the hot path — ~8K multiply-adds.
    #[inline]
    pub fn evaluate_from_accumulator(&self, acc: &NnueAccumulator) -> f32 {
        // Layer 2: W2 * ReLU(accumulator) + b2, then ReLU
        // Fused ReLU: apply max(0, hidden[j]) inline instead of separate pass.
        let mut l2_out = [0.0f32; L2_SIZE];
        for i in 0..L2_SIZE {
            let mut sum = self.weights.l2_bias[i];
            let row = &self.weights.l2_weight[i * L1_SIZE..(i + 1) * L1_SIZE];
            for j in 0..L1_SIZE {
                sum += row[j] * acc.hidden[j].max(0.0);
            }
            l2_out[i] = sum.max(0.0);
        }

        // Layer 3: W3 * l2_out + b3, then Sigmoid
        let mut output = self.weights.l3_bias[0];
        for j in 0..L2_SIZE {
            output += self.weights.l3_weight[j] * l2_out[j];
        }

        1.0 / (1.0 + (-output).exp())
    }

    pub fn weights(&self) -> &NnueWeights {
        &self.weights
    }
}
