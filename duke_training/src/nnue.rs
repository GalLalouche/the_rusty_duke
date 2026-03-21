use duke_rust::game::state::GameState;
use crate::encoding::{active_feature_indices, NUM_PLANES, BOARD_SIZE};

pub const NUM_FEATURES: usize = NUM_PLANES * BOARD_SIZE * BOARD_SIZE; // 1080
pub const L1_SIZE: usize = 256;
pub const L2_SIZE: usize = 32;

/// Raw model weights for NNUE inference.
pub struct NnueWeights {
    /// Layer 1 weights [L1_SIZE x NUM_FEATURES], row-major.
    /// l1_weight[i * NUM_FEATURES + j] = weight from input j to hidden unit i.
    pub l1_weight: Vec<f32>,
    /// Layer 1 bias [L1_SIZE].
    pub l1_bias: Vec<f32>,
    /// Layer 2 weights [L2_SIZE x L1_SIZE], row-major.
    pub l2_weight: Vec<f32>,
    /// Layer 2 bias [L2_SIZE].
    pub l2_bias: Vec<f32>,
    /// Layer 3 weights [1 x L2_SIZE], row-major.
    pub l3_weight: Vec<f32>,
    /// Layer 3 bias [1].
    pub l3_bias: Vec<f32>,
}

impl NnueWeights {
    const MAGIC: &'static [u8; 4] = b"DUKE";
    const VERSION: u32 = 1;

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
            "Unsupported NNUE version"
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
            l1_weight: read_vec(&mut f, L1_SIZE * NUM_FEATURES)?,
            l1_bias: read_vec(&mut f, L1_SIZE)?,
            l2_weight: read_vec(&mut f, L2_SIZE * L1_SIZE)?,
            l2_bias: read_vec(&mut f, L2_SIZE)?,
            l3_weight: read_vec(&mut f, L2_SIZE)?, // 1 * L2_SIZE
            l3_bias: read_vec(&mut f, 1)?,
        })
    }
}

/// Cached first-layer output (pre-ReLU). 256 floats = 1KB.
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
        // Start with bias
        hidden.copy_from_slice(&weights.l1_bias);
        // Add columns for active features
        for &feat in features {
            debug_assert!(feat < NUM_FEATURES);
            for i in 0..L1_SIZE {
                hidden[i] += weights.l1_weight[i * NUM_FEATURES + feat];
            }
        }
        Self { hidden }
    }

    /// Add a feature (tile placed/moved to a square).
    pub fn add_feature(&mut self, feat: usize, weights: &NnueWeights) {
        for i in 0..L1_SIZE {
            self.hidden[i] += weights.l1_weight[i * NUM_FEATURES + feat];
        }
    }

    /// Remove a feature (tile removed/moved from a square).
    pub fn remove_feature(&mut self, feat: usize, weights: &NnueWeights) {
        for i in 0..L1_SIZE {
            self.hidden[i] -= weights.l1_weight[i * NUM_FEATURES + feat];
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

    /// Evaluate a game state from scratch.
    /// Returns win probability for the current player in [0, 1].
    pub fn evaluate_state(&self, gs: &GameState) -> f32 {
        let features = active_feature_indices(gs);
        let acc = NnueAccumulator::from_features(&self.weights, &features);
        self.evaluate_from_accumulator(&acc)
    }

    /// Evaluate from a pre-computed accumulator (layers 2-3 only).
    /// This is the hot path -- ~8K multiply-adds.
    pub fn evaluate_from_accumulator(&self, acc: &NnueAccumulator) -> f32 {
        // Layer 1 output: ReLU(accumulator)
        let mut l1_out = [0.0f32; L1_SIZE];
        for i in 0..L1_SIZE {
            l1_out[i] = acc.hidden[i].max(0.0);
        }

        // Layer 2: W2 * l1_out + b2, then ReLU
        let mut l2_out = [0.0f32; L2_SIZE];
        for i in 0..L2_SIZE {
            let mut sum = self.weights.l2_bias[i];
            for j in 0..L1_SIZE {
                sum += self.weights.l2_weight[i * L1_SIZE + j] * l1_out[j];
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
