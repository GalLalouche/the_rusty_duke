use duke_rust::game::state::GameState;
use crate::encoding::{active_feature_indices, TOTAL_FEATURES, BOARD_FEATURES, bag_features};

pub const NUM_FEATURES: usize = TOTAL_FEATURES;
/// Default sizes — can be overridden at load time or by `export_weights`.
pub const DEFAULT_L1: usize = 256;
pub const DEFAULT_L2: usize = 32;

/// Maximum supported layer sizes for stack allocation.
const MAX_L1: usize = 1024;
const MAX_L2: usize = 128;

/// Raw model weights for NNUE inference.
///
/// L1 weights are column-major: l1_weight[feat * l1_size + i].
pub struct NnueWeights {
    pub l1_size: usize,
    pub l2_size: usize,
    pub l1_weight: Vec<f32>,
    pub l1_bias: Vec<f32>,
    pub l2_weight: Vec<f32>,
    pub l2_bias: Vec<f32>,
    pub l3_weight: Vec<f32>,
    pub l3_bias: Vec<f32>,
}

impl NnueWeights {
    const MAGIC: &'static [u8; 4] = b"DUKE";
    const VERSION: u32 = 3; // Bumped: stores layer sizes in file

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(Self::MAGIC)?;
        f.write_all(&Self::VERSION.to_le_bytes())?;
        f.write_all(&(self.l1_size as u32).to_le_bytes())?;
        f.write_all(&(self.l2_size as u32).to_le_bytes())?;
        for v in [
            &self.l1_weight, &self.l1_bias,
            &self.l2_weight, &self.l2_bias,
            &self.l3_weight, &self.l3_bias,
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
        if &magic != Self::MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid NNUE file magic"));
        }

        let mut version = [0u8; 4];
        f.read_exact(&mut version)?;
        let version = u32::from_le_bytes(version);
        if version != 3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported NNUE version (expected 3, got {})", version),
            ));
        }

        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let l1_size = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let l2_size = u32::from_le_bytes(buf4) as usize;

        if l1_size > MAX_L1 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("l1_size {} exceeds MAX_L1 {}", l1_size, MAX_L1)));
        }
        if l2_size > MAX_L2 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("l2_size {} exceeds MAX_L2 {}", l2_size, MAX_L2)));
        }

        let read_vec = |f: &mut std::fs::File, n: usize| -> std::io::Result<Vec<f32>> {
            let mut buf = vec![0u8; n * 4];
            f.read_exact(&mut buf)?;
            Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        };

        Ok(Self {
            l1_size,
            l2_size,
            l1_weight: read_vec(&mut f, NUM_FEATURES * l1_size)?,
            l1_bias: read_vec(&mut f, l1_size)?,
            l2_weight: read_vec(&mut f, l2_size * l1_size)?,
            l2_bias: read_vec(&mut f, l2_size)?,
            l3_weight: read_vec(&mut f, l2_size)?,
            l3_bias: read_vec(&mut f, 1)?,
        })
    }
}

#[derive(Clone)]
pub struct NnueAccumulator {
    pub hidden: [f32; MAX_L1],
    l1_size: usize,
}

impl NnueAccumulator {
    pub fn from_features(weights: &NnueWeights, features: &[usize]) -> Self {
        let l1 = weights.l1_size;
        assert!(l1 <= MAX_L1);
        let mut hidden = [0.0f32; MAX_L1];
        hidden[..l1].copy_from_slice(&weights.l1_bias);
        for &feat in features {
            let col = &weights.l1_weight[feat * l1..(feat + 1) * l1];
            for i in 0..l1 {
                hidden[i] += col[i];
            }
        }
        Self { hidden, l1_size: l1 }
    }

    /// Incrementally add a binary (0/1) board feature.
    /// Only valid for board features (index < BOARD_FEATURES); bag features
    /// are non-binary and must not be updated through this method.
    #[inline]
    pub fn add_feature(&mut self, feat: usize, weights: &NnueWeights) {
        debug_assert!(feat < BOARD_FEATURES, "add_feature called with bag feature index {}", feat);
        let l1 = self.l1_size;
        let col = &weights.l1_weight[feat * l1..(feat + 1) * l1];
        for i in 0..l1 {
            self.hidden[i] += col[i];
        }
    }

    #[inline]
    pub fn remove_feature(&mut self, feat: usize, weights: &NnueWeights) {
        debug_assert!(feat < BOARD_FEATURES, "remove_feature called with bag feature index {}", feat);
        let l1 = self.l1_size;
        let col = &weights.l1_weight[feat * l1..(feat + 1) * l1];
        for i in 0..l1 {
            self.hidden[i] -= col[i];
        }
    }
}

pub struct NnueEvaluator {
    weights: NnueWeights,
}

impl NnueEvaluator {
    pub fn new(weights: NnueWeights) -> Self {
        Self { weights }
    }

    pub fn evaluate_state(&self, gs: &GameState) -> f32 {
        let board_features = active_feature_indices(gs);
        let mut acc = NnueAccumulator::from_features(&self.weights, board_features.as_slice());

        let bag = bag_features(gs);
        let l1 = self.weights.l1_size;
        for (i, &val) in bag.iter().enumerate() {
            if val != 0.0 {
                let feat = BOARD_FEATURES + i;
                let col = &self.weights.l1_weight[feat * l1..(feat + 1) * l1];
                for j in 0..l1 {
                    acc.hidden[j] += col[j] * val;
                }
            }
        }

        self.evaluate_from_accumulator(&acc)
    }

    #[inline]
    pub fn evaluate_from_accumulator(&self, acc: &NnueAccumulator) -> f32 {
        let l1 = self.weights.l1_size;
        let l2 = self.weights.l2_size;

        // L2: W2 * ReLU(acc) + b2, then ReLU
        assert!(l2 <= MAX_L2, "l2_size {} exceeds MAX_L2 {}", l2, MAX_L2);
        let mut l2_out = [0.0f32; MAX_L2];
        for i in 0..l2 {
            let mut sum = self.weights.l2_bias[i];
            let row = &self.weights.l2_weight[i * l1..(i + 1) * l1];
            for j in 0..l1 {
                sum += row[j] * acc.hidden[j].max(0.0);
            }
            l2_out[i] = sum.max(0.0);
        }

        // L3: W3 * l2_out + b3, then Sigmoid
        let mut output = self.weights.l3_bias[0];
        for j in 0..l2 {
            output += self.weights.l3_weight[j] * l2_out[j];
        }

        1.0 / (1.0 + (-output).exp())
    }

    pub fn weights(&self) -> &NnueWeights { &self.weights }
}
