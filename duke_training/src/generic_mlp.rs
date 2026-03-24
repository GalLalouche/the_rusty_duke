//! Generic MLP network and evaluator wrappers, shared between es_train and elo_tournament.

use rand::rngs::StdRng;
use rand::Rng;

use duke_rust::game::state::GameState;

use crate::encoding::{active_board_features, bag_features, BOARD_FEATURES, TOTAL_FEATURES};
use crate::game_setup::{GameEvaluator, StaticHeuristicEvaluator};
use crate::learned_heuristic::{extract_combined_features, extract_features, NUM_COMBINED_FEATURES};
use crate::nnue::{NnueEvaluator, NnueWeights, NUM_FEATURES};

// ── Generic MLP network (N hidden layers) ────────────────────────────────
// A self-contained input->H1->H2->...->Hn->1 network with ReLU hidden layers
// and sigmoid output. Operates entirely on Vec<f32>.

/// Maximum hidden layer size for stack allocation in forward passes.
pub const MAX_HIDDEN: usize = 1024;

/// Total guard feature count: 24 expensive + 41 combined = 65.
pub const NUM_GUARD_ALL_FEATURES: usize = 24 + NUM_COMBINED_FEATURES;

/// Generic MLP weight container supporting arbitrary hidden layer depths.
pub struct GenericMlp {
    pub input_size: usize,
    /// Hidden layer sizes, e.g. [64, 64, 32] for 3 hidden layers
    pub hidden_layers: Vec<usize>,
    /// Flat weight vector: [h1_w, h1_b, h2_w, h2_b, ..., out_w, out_b]
    pub weights: Vec<f32>,
}

impl GenericMlp {
    /// Compute total parameter count for input_size -> hidden_layers -> 1.
    pub fn param_count(input_size: usize, hidden_layers: &[usize]) -> usize {
        assert!(!hidden_layers.is_empty(), "Need at least one hidden layer");
        let mut count = 0;
        let mut prev = input_size;
        for &h in hidden_layers {
            count += prev * h + h; // weight + bias
            prev = h;
        }
        count += prev + 1; // output weight + bias
        count
    }

    pub fn from_flat(flat: Vec<f32>, input_size: usize, hidden_layers: Vec<usize>) -> Self {
        let expected = Self::param_count(input_size, &hidden_layers);
        assert_eq!(
            flat.len(), expected,
            "flat weight vector size mismatch: expected {}, got {}",
            expected, flat.len()
        );
        for &h in &hidden_layers {
            assert!(h <= MAX_HIDDEN, "hidden layer size {} exceeds MAX_HIDDEN {}", h, MAX_HIDDEN);
        }
        Self { input_size, hidden_layers, weights: flat }
    }

    pub fn random(input_size: usize, hidden_layers: Vec<usize>, rng: &mut StdRng) -> Self {
        let n = Self::param_count(input_size, &hidden_layers);
        let mut flat = Vec::with_capacity(n);

        let mut prev = input_size;
        for &h in &hidden_layers {
            // Kaiming uniform init: U(-sqrt(6/fan_in), +sqrt(6/fan_in))
            let scale = (6.0 / prev as f64).sqrt() as f32;
            for _ in 0..(prev * h) {
                flat.push(rng.gen::<f32>() * 2.0 * scale - scale);
            }
            // Bias = 0
            for _ in 0..h { flat.push(0.0); }
            prev = h;
        }

        // Output layer weights: Kaiming uniform, fan_in = last hidden
        let scale_out = (6.0 / prev as f64).sqrt() as f32;
        for _ in 0..prev {
            flat.push(rng.gen::<f32>() * 2.0 * scale_out - scale_out);
        }
        // Output bias
        flat.push(0.0);

        assert_eq!(flat.len(), n);
        Self { input_size, hidden_layers, weights: flat }
    }

    /// Forward pass with f64 input (for combined features): input -> H1(ReLU) -> ... -> sigmoid
    pub fn forward_f64(&self, input: &[f64]) -> f32 {
        assert_eq!(input.len(), self.input_size);
        let w = &self.weights;
        let mut off = 0;

        // First hidden layer: f64 input -> f32
        let h_size = self.hidden_layers[0];
        let hw = &w[off..off + self.input_size * h_size];
        off += self.input_size * h_size;
        let hb = &w[off..off + h_size];
        off += h_size;

        // Use two stack buffers and ping-pong between them (no heap allocation).
        let mut buf_a = [0.0f32; MAX_HIDDEN];
        let mut buf_b = [0.0f32; MAX_HIDDEN];
        let mut use_a = true; // buf_a holds the current layer's activations

        for j in 0..h_size {
            let mut sum = hb[j];
            for i in 0..self.input_size {
                sum += hw[i * h_size + j] * input[i] as f32;
            }
            buf_a[j] = sum.max(0.0); // ReLU
        }

        // Subsequent hidden layers: f32 -> f32
        for layer_idx in 1..self.hidden_layers.len() {
            let prev_size = self.hidden_layers[layer_idx - 1];
            let cur_size = self.hidden_layers[layer_idx];
            let lw = &w[off..off + prev_size * cur_size];
            off += prev_size * cur_size;
            let lb = &w[off..off + cur_size];
            off += cur_size;

            let (src, dst) = if use_a { (&buf_a, &mut buf_b) } else { (&buf_b, &mut buf_a) };
            for j in 0..cur_size {
                let mut sum = lb[j];
                for i in 0..prev_size {
                    sum += lw[i * cur_size + j] * src[i];
                }
                dst[j] = sum.max(0.0); // ReLU
            }
            use_a = !use_a;
        }

        // Output layer: last_hidden -> 1, sigmoid
        let last_h = *self.hidden_layers.last().unwrap();
        let out_w = &w[off..off + last_h];
        off += last_h;
        let out_b = w[off];

        let prev = if use_a { &buf_a } else { &buf_b };
        let mut logit = out_b;
        for i in 0..last_h {
            logit += out_w[i] * prev[i];
        }

        1.0 / (1.0 + (-logit).exp())
    }

    /// Forward pass with f32 input: input -> H1(ReLU) -> ... -> sigmoid
    pub fn forward_f32(&self, input: &[f32]) -> f32 {
        assert_eq!(input.len(), self.input_size);
        let w = &self.weights;
        let mut off = 0;

        let mut buf_a = [0.0f32; MAX_HIDDEN];
        let mut buf_b = [0.0f32; MAX_HIDDEN];
        let mut use_a = true;

        // First hidden layer
        let h_size = self.hidden_layers[0];
        let hw = &w[off..off + self.input_size * h_size];
        off += self.input_size * h_size;
        let hb = &w[off..off + h_size];
        off += h_size;

        for j in 0..h_size {
            let mut sum = hb[j];
            for i in 0..self.input_size {
                sum += hw[i * h_size + j] * input[i];
            }
            buf_a[j] = sum.max(0.0);
        }

        // Subsequent hidden layers
        for layer_idx in 1..self.hidden_layers.len() {
            let prev_size = self.hidden_layers[layer_idx - 1];
            let cur_size = self.hidden_layers[layer_idx];
            let lw = &w[off..off + prev_size * cur_size];
            off += prev_size * cur_size;
            let lb = &w[off..off + cur_size];
            off += cur_size;

            let (src, dst) = if use_a { (&buf_a, &mut buf_b) } else { (&buf_b, &mut buf_a) };
            for j in 0..cur_size {
                let mut sum = lb[j];
                for i in 0..prev_size {
                    sum += lw[i * cur_size + j] * src[i];
                }
                dst[j] = sum.max(0.0);
            }
            use_a = !use_a;
        }

        // Output layer
        let last_h = *self.hidden_layers.last().unwrap();
        let out_w = &w[off..off + last_h];
        off += last_h;
        let out_b = w[off];

        let prev = if use_a { &buf_a } else { &buf_b };
        let mut logit = out_b;
        for i in 0..last_h {
            logit += out_w[i] * prev[i];
        }

        1.0 / (1.0 + (-logit).exp())
    }

    /// Forward pass optimized for sparse NNUE-style inputs (1106 or 1147 dims).
    /// Uses active_board_features for sparse binary features,
    /// bag_features for dense bag dims, and optionally combined features.
    pub fn forward_sparse(&self, gs: &GameState, include_combined: bool) -> f32 {
        let w = &self.weights;
        let h1 = self.hidden_layers[0];

        // L1: sparse accumulation
        let l1_w = &w[0..self.input_size * h1];
        let l1_b = &w[self.input_size * h1..self.input_size * h1 + h1];
        let mut off = self.input_size * h1 + h1;

        let mut buf_a = [0.0f32; MAX_HIDDEN];
        buf_a[..h1].copy_from_slice(l1_b);

        // Sparse board features (binary)
        let board_feats = active_board_features(gs);
        for &feat in board_feats.as_slice() {
            let col = &l1_w[feat * h1..(feat + 1) * h1];
            for j in 0..h1 {
                buf_a[j] += col[j];
            }
        }

        // Bag features (dense, dimensions 1080..1106)
        let bag = bag_features(gs);
        for (i, &val) in bag.iter().enumerate() {
            if val != 0.0 {
                let feat = BOARD_FEATURES + i;
                let col = &l1_w[feat * h1..(feat + 1) * h1];
                for j in 0..h1 {
                    buf_a[j] += col[j] * val;
                }
            }
        }

        // Combined features (if appended mode, dimensions 1106..1147)
        if include_combined {
            let combined = extract_combined_features(gs);
            for (i, &val) in combined.iter().enumerate() {
                let fval = val as f32;
                if fval != 0.0 {
                    let feat = TOTAL_FEATURES + i;
                    let col = &l1_w[feat * h1..(feat + 1) * h1];
                    if fval == 1.0 {
                        for j in 0..h1 {
                            buf_a[j] += col[j];
                        }
                    } else {
                        for j in 0..h1 {
                            buf_a[j] += col[j] * fval;
                        }
                    }
                }
            }
        }

        // ReLU
        for j in 0..h1 {
            buf_a[j] = buf_a[j].max(0.0);
        }

        // Subsequent hidden layers (ping-pong between buf_a and buf_b)
        let mut buf_b = [0.0f32; MAX_HIDDEN];
        let mut use_a = true;
        for layer_idx in 1..self.hidden_layers.len() {
            let prev_size = self.hidden_layers[layer_idx - 1];
            let cur_size = self.hidden_layers[layer_idx];
            let lw = &w[off..off + prev_size * cur_size];
            off += prev_size * cur_size;
            let lb = &w[off..off + cur_size];
            off += cur_size;

            let (src, dst) = if use_a { (&buf_a, &mut buf_b) } else { (&buf_b, &mut buf_a) };
            for j in 0..cur_size {
                let mut sum = lb[j];
                for i in 0..prev_size {
                    sum += lw[i * cur_size + j] * src[i];
                }
                dst[j] = sum.max(0.0);
            }
            use_a = !use_a;
        }

        // Output layer
        let last_h = *self.hidden_layers.last().unwrap();
        let out_w = &w[off..off + last_h];
        off += last_h;
        let out_b = w[off];

        let prev = if use_a { &buf_a } else { &buf_b };
        let mut logit = out_b;
        for i in 0..last_h {
            logit += out_w[i] * prev[i];
        }

        1.0 / (1.0 + (-logit).exp())
    }

    /// Format the architecture as a string like "1106->64->64->32->1"
    pub fn arch_string(&self) -> String {
        let mut s = format!("{}", self.input_size);
        for &h in &self.hidden_layers {
            s.push_str(&format!("->{}", h));
        }
        s.push_str("->1");
        s
    }

    /// Save weights as a binary file: [magic "GMLP", version, num_layers, input_size, h1, h2, ..., f32 weights...]
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"GMLP")?; // magic for Generic MLP
        f.write_all(&1u32.to_le_bytes())?; // version
        f.write_all(&(self.hidden_layers.len() as u32).to_le_bytes())?;
        f.write_all(&(self.input_size as u32).to_le_bytes())?;
        for &h in &self.hidden_layers {
            f.write_all(&(h as u32).to_le_bytes())?;
        }
        for &val in &self.weights {
            f.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load weights from a binary file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"GMLP" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Not a GMLP file",
            ));
        }
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("Unsupported GMLP version {}", version)));
        }
        f.read_exact(&mut buf4)?;
        let num_layers = u32::from_le_bytes(buf4) as usize;
        if num_layers == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "num_layers must be > 0",
            ));
        }
        f.read_exact(&mut buf4)?;
        let input_size = u32::from_le_bytes(buf4) as usize;
        let mut hidden_layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            f.read_exact(&mut buf4)?;
            hidden_layers.push(u32::from_le_bytes(buf4) as usize);
        }

        for &h in &hidden_layers {
            if h > MAX_HIDDEN {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("hidden layer size {} exceeds MAX_HIDDEN {}", h, MAX_HIDDEN)));
            }
        }
        let n = Self::param_count(input_size, &hidden_layers);
        let mut weights = vec![0.0f32; n];
        for val in &mut weights {
            f.read_exact(&mut buf4)?;
            *val = f32::from_le_bytes(buf4);
        }
        Ok(Self { input_size, hidden_layers, weights })
    }
}

/// Evaluator that wraps a GenericMlp: extracts combined features then forward-passes.
pub struct CombinedNetEvaluator {
    pub net: GenericMlp,
}

impl CombinedNetEvaluator {
    pub fn new(net: GenericMlp) -> Self {
        Self { net }
    }
}

impl GameEvaluator for CombinedNetEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        let features = extract_combined_features(gs);
        self.net.forward_f64(&features)
    }
}

/// Evaluator that wraps a GenericMlp: extracts all 65 features (24 expensive + 41 combined)
/// then forward-passes. This is the richest feature set with the most signal.
pub struct GuardFeatureEvaluator {
    pub net: GenericMlp,
}

impl GuardFeatureEvaluator {
    pub fn new(net: GenericMlp) -> Self {
        Self { net }
    }
}

impl GameEvaluator for GuardFeatureEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        let expensive = extract_features(gs);       // 24 values
        let combined = extract_combined_features(gs); // 41 values
        let mut features = [0.0f64; NUM_GUARD_ALL_FEATURES];
        features[..24].copy_from_slice(&expensive);
        features[24..].copy_from_slice(&combined);
        self.net.forward_f64(&features)
    }
}

/// Evaluator that wraps a GenericMlp for sparse NNUE features (1106 inputs).
pub struct GenericNnueEvaluator {
    pub net: GenericMlp,
}

impl GameEvaluator for GenericNnueEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        self.net.forward_sparse(gs, false)
    }
}

/// Evaluator that wraps a GenericMlp for appended features (1147 inputs).
pub struct GenericAppendedEvaluator {
    pub net: GenericMlp,
}

impl GameEvaluator for GenericAppendedEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        self.net.forward_sparse(gs, true)
    }
}

/// Load an opponent evaluator from a file path or keyword.
///
/// Returns `None` for the "random" keyword (caller should use `Player::Random`),
/// or `Some(evaluator)` for all other cases.
///
/// Supported formats:
///   - "base" or absent  -> StaticHeuristicEvaluator
///   - "random"          -> None (caller uses Player::Random)
///   - path ending .gmlp -> GenericMlp, dispatched by input_size
///   - path ending .nnue -> NnueWeights wrapped in NnueEvaluator
pub fn load_opponent(spec: &str) -> (Option<Box<dyn GameEvaluator + Sync + Send>>, String) {
    match spec {
        "base" => {
            let eval = StaticHeuristicEvaluator::new();
            (Some(Box::new(eval)), "Base heuristic".to_string())
        }
        "random" => {
            (None, "Random".to_string())
        }
        path if path.ends_with(".gmlp") => {
            let net = GenericMlp::load(path).expect("Failed to load .gmlp opponent");
            let desc = format!("GMLP ({})", net.arch_string());
            let eval: Box<dyn GameEvaluator + Sync + Send> = match net.input_size {
                41 => Box::new(CombinedNetEvaluator::new(net)),
                65 => Box::new(GuardFeatureEvaluator::new(net)),
                1106 => Box::new(GenericNnueEvaluator { net }),
                1147 => Box::new(GenericAppendedEvaluator { net }),
                other => panic!(
                    "Unknown input_size {} in .gmlp file '{}'. Expected 41, 65, 1106, or 1147.",
                    other, path
                ),
            };
            (Some(eval), desc)
        }
        path if path.ends_with(".nnue") => {
            let weights = NnueWeights::load(path).expect("Failed to load .nnue opponent");
            let desc = format!("NNUE ({}->{}->{}->1)", NUM_FEATURES, weights.l1_size, weights.l2_size);
            let eval = NnueEvaluator::new(weights);
            (Some(Box::new(eval)), desc)
        }
        other => {
            panic!(
                "Unknown opponent '{}'. Use 'base', 'random', or a path ending in .gmlp / .nnue",
                other
            );
        }
    }
}
