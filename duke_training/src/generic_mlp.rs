//! Generic MLP network and evaluator wrappers, shared between es_train and elo_tournament.

use rand::rngs::StdRng;
use rand::Rng;

use duke_rust::game::state::GameState;

use crate::encoding::{
    active_board_features, bag_features, FeatureBuffer, BOARD_FEATURES, BAG_FEATURES,
    TOTAL_FEATURES,
};
use crate::game_setup::{GameEvaluator, StaticHeuristicEvaluator};
use crate::learned_heuristic::{
    extract_combined_features, extract_features, load_lr_weights_raw,
    AllFeaturesWeights, CombinedWeights, LearnedHeuristicWeights,
    NUM_ALL_FEATURES, NUM_COMBINED_FEATURES, NUM_FEATURES as LR_NUM_FEATURES,
};
use crate::nnue::{NnueEvaluator, NnueWeights, NUM_FEATURES};

// ── Generic MLP network (N hidden layers) ────────────────────────────────
// A self-contained input->H1->H2->...->Hn->1 network with ReLU hidden layers
// and sigmoid output. Operates entirely on Vec<f32>.

/// Maximum hidden layer size for stack allocation in forward passes.
pub const MAX_HIDDEN: usize = 1024;

/// Total guard feature count: 24 expensive + 41 combined = 65.
pub const NUM_GUARD_ALL_FEATURES: usize = 24 + NUM_COMBINED_FEATURES;

/// Generic MLP weight container supporting arbitrary hidden layer depths.
#[derive(Debug)]
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

    /// Shared helper: propagate through hidden layers 2..N and the output sigmoid.
    ///
    /// `buf_a` must already contain the ReLU-activated first hidden layer output.
    /// `buf_b` is scratch space. Both are stack-allocated `[f32; MAX_HIDDEN]` ping-pong
    /// buffers. `off` is the current offset into `self.weights`, pointing just past
    /// the first hidden layer's weights+biases.
    fn forward_inner(
        &self,
        buf_a: &mut [f32; MAX_HIDDEN],
        buf_b: &mut [f32; MAX_HIDDEN],
        mut off: usize,
    ) -> f32 {
        let w = &self.weights;
        let mut use_a = true; // buf_a holds the current activations

        // Subsequent hidden layers: f32 -> f32
        for layer_idx in 1..self.hidden_layers.len() {
            let prev_size = self.hidden_layers[layer_idx - 1];
            let cur_size = self.hidden_layers[layer_idx];
            let lw = &w[off..off + prev_size * cur_size];
            off += prev_size * cur_size;
            let lb = &w[off..off + cur_size];
            off += cur_size;

            let (src, dst) = if use_a {
                (&*buf_a as &[f32; MAX_HIDDEN], &mut *buf_b)
            } else {
                (&*buf_b as &[f32; MAX_HIDDEN], &mut *buf_a)
            };
            // Initialize with bias, then scatter-accumulate (contiguous weight access)
            dst[..cur_size].copy_from_slice(lb);
            for i in 0..prev_size {
                let w_row = &lw[i * cur_size..(i + 1) * cur_size];
                let s = src[i];
                for j in 0..cur_size {
                    dst[j] += w_row[j] * s;
                }
            }
            for j in 0..cur_size {
                dst[j] = dst[j].max(0.0); // ReLU
            }
            use_a = !use_a;
        }

        // Output layer: last_hidden -> 1, sigmoid
        let last_h = *self.hidden_layers.last().unwrap();
        let out_w = &w[off..off + last_h];
        off += last_h;
        let out_b = w[off];

        let prev = if use_a { &*buf_a } else { &*buf_b };
        let mut logit = out_b;
        for i in 0..last_h {
            logit += out_w[i] * prev[i];
        }

        1.0 / (1.0 + (-logit).exp())
    }

    /// Forward pass with f64 input (for combined features): input -> H1(ReLU) -> ... -> sigmoid
    pub fn forward_f64(&self, input: &[f64]) -> f32 {
        assert_eq!(input.len(), self.input_size);
        let w = &self.weights;

        // First hidden layer: f64 input -> f32
        let h_size = self.hidden_layers[0];
        let hw = &w[0..self.input_size * h_size];
        let hb = &w[self.input_size * h_size..self.input_size * h_size + h_size];
        let off = self.input_size * h_size + h_size;

        let mut buf_a = [0.0f32; MAX_HIDDEN];
        let mut buf_b = [0.0f32; MAX_HIDDEN];

        // Initialize with bias, then scatter-accumulate (contiguous weight access)
        buf_a[..h_size].copy_from_slice(hb);
        for i in 0..self.input_size {
            let w_row = &hw[i * h_size..(i + 1) * h_size];
            let inp = input[i] as f32;
            for j in 0..h_size {
                buf_a[j] += w_row[j] * inp;
            }
        }
        for j in 0..h_size {
            buf_a[j] = buf_a[j].max(0.0); // ReLU
        }

        self.forward_inner(&mut buf_a, &mut buf_b, off)
    }

    /// Forward pass with f32 input: input -> H1(ReLU) -> ... -> sigmoid
    pub fn forward_f32(&self, input: &[f32]) -> f32 {
        assert_eq!(input.len(), self.input_size);
        let w = &self.weights;

        // First hidden layer
        let h_size = self.hidden_layers[0];
        let hw = &w[0..self.input_size * h_size];
        let hb = &w[self.input_size * h_size..self.input_size * h_size + h_size];
        let off = self.input_size * h_size + h_size;

        let mut buf_a = [0.0f32; MAX_HIDDEN];
        let mut buf_b = [0.0f32; MAX_HIDDEN];

        // Initialize with bias, then scatter-accumulate (contiguous weight access)
        buf_a[..h_size].copy_from_slice(hb);
        for i in 0..self.input_size {
            let w_row = &hw[i * h_size..(i + 1) * h_size];
            let inp = input[i];
            for j in 0..h_size {
                buf_a[j] += w_row[j] * inp;
            }
        }
        for j in 0..h_size {
            buf_a[j] = buf_a[j].max(0.0); // ReLU
        }

        self.forward_inner(&mut buf_a, &mut buf_b, off)
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
        let off = self.input_size * h1 + h1;

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

        let mut buf_b = [0.0f32; MAX_HIDDEN];
        self.forward_inner(&mut buf_a, &mut buf_b, off)
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

// ── L1 Accumulator for incremental updates ──────────────────────────────

/// Pre-computed first hidden layer (L1) activations for incremental updates.
///
/// Stores the raw accumulation BEFORE ReLU (bias + weighted sum of active
/// features).  ReLU is applied lazily when completing the forward pass via
/// [`L1Accumulator::forward`].
///
/// Typical workflow in move search:
/// 1. Build an accumulator from the current position (`from_state`).
/// 2. For each candidate move, clone the accumulator, apply the feature
///    diff (`update_features`), then call `forward` to get the evaluation.
///
/// This avoids redundant L1 recomputation across candidates — only the
/// 2-4 changed features need to be patched instead of all ~24.
pub struct L1Accumulator {
    /// Raw L1 values (bias + weighted sum of active features), before ReLU.
    hidden: [f32; MAX_HIDDEN],
    /// Number of active hidden units (= hidden_layers[0]).
    h1_size: usize,
}

impl Clone for L1Accumulator {
    fn clone(&self) -> Self {
        Self {
            hidden: self.hidden,
            h1_size: self.h1_size,
        }
    }
}

impl L1Accumulator {
    /// Build an L1 accumulator from a game state (full recompute).
    ///
    /// Performs the same sparse accumulation as `forward_sparse`, but stops
    /// before applying ReLU, storing the raw weighted sums.
    pub fn from_state(net: &GenericMlp, gs: &GameState, include_combined: bool) -> Self {
        let h1 = net.hidden_layers[0];
        let w = &net.weights;
        let l1_w = &w[0..net.input_size * h1];
        let l1_b = &w[net.input_size * h1..net.input_size * h1 + h1];

        let mut hidden = [0.0f32; MAX_HIDDEN];
        hidden[..h1].copy_from_slice(l1_b);

        // Sparse board features (binary)
        let board_feats = active_board_features(gs);
        for &feat in board_feats.as_slice() {
            let col = &l1_w[feat * h1..(feat + 1) * h1];
            for j in 0..h1 {
                hidden[j] += col[j];
            }
        }

        // Bag features (dense, dimensions 1080..1106)
        let bag = bag_features(gs);
        for (i, &val) in bag.iter().enumerate() {
            if val != 0.0 {
                let feat = BOARD_FEATURES + i;
                let col = &l1_w[feat * h1..(feat + 1) * h1];
                for j in 0..h1 {
                    hidden[j] += col[j] * val;
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
                            hidden[j] += col[j];
                        }
                    } else {
                        for j in 0..h1 {
                            hidden[j] += col[j] * fval;
                        }
                    }
                }
            }
        }

        Self { hidden, h1_size: h1 }
    }

    /// Update this accumulator to reflect a feature diff.
    ///
    /// Given the old and new board features, bag features, and optionally
    /// combined features, patches the raw L1 values by subtracting removed
    /// contributions and adding new ones.
    ///
    /// This is much cheaper than a full rebuild when only 2-4 features change
    /// (the common case for a single move).
    pub fn update_features(
        &mut self,
        net: &GenericMlp,
        old_board: &FeatureBuffer,
        new_board: &FeatureBuffer,
        old_bag: &[f32; BAG_FEATURES],
        new_bag: &[f32; BAG_FEATURES],
        old_combined: Option<&[f64; NUM_COMBINED_FEATURES]>,
        new_combined: Option<&[f64; NUM_COMBINED_FEATURES]>,
    ) {
        let h1 = self.h1_size;
        let l1_w = &net.weights[0..net.input_size * h1];

        // --- Board features diff (binary: just add/remove weight rows) ---
        // Remove old board features not in new set
        for &feat in old_board.as_slice() {
            if !new_board.as_slice().contains(&feat) {
                let col = &l1_w[feat * h1..(feat + 1) * h1];
                for j in 0..h1 {
                    self.hidden[j] -= col[j];
                }
            }
        }
        // Add new board features not in old set
        for &feat in new_board.as_slice() {
            if !old_board.as_slice().contains(&feat) {
                let col = &l1_w[feat * h1..(feat + 1) * h1];
                for j in 0..h1 {
                    self.hidden[j] += col[j];
                }
            }
        }

        // --- Bag features diff (dense: subtract old, add new for changed dims) ---
        for i in 0..BAG_FEATURES {
            let delta = new_bag[i] - old_bag[i];
            if delta != 0.0 {
                let feat = BOARD_FEATURES + i;
                let col = &l1_w[feat * h1..(feat + 1) * h1];
                for j in 0..h1 {
                    self.hidden[j] += col[j] * delta;
                }
            }
        }

        // --- Combined features diff ---
        if let (Some(old_c), Some(new_c)) = (old_combined, new_combined) {
            for i in 0..NUM_COMBINED_FEATURES {
                let delta = (new_c[i] - old_c[i]) as f32;
                if delta != 0.0 {
                    let feat = TOTAL_FEATURES + i;
                    let col = &l1_w[feat * h1..(feat + 1) * h1];
                    for j in 0..h1 {
                        self.hidden[j] += col[j] * delta;
                    }
                }
            }
        }
    }

    /// Complete the forward pass: apply ReLU to the stored L1 values, then
    /// propagate through the remaining hidden layers and output sigmoid.
    pub fn forward(&self, net: &GenericMlp) -> f32 {
        let h1 = self.h1_size;
        let off = net.input_size * h1 + h1;

        // Copy hidden into buf_a and apply ReLU
        let mut buf_a = [0.0f32; MAX_HIDDEN];
        for j in 0..h1 {
            buf_a[j] = self.hidden[j].max(0.0);
        }

        let mut buf_b = [0.0f32; MAX_HIDDEN];
        net.forward_inner(&mut buf_a, &mut buf_b, off)
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
    fn as_generic_mlp(&self) -> Option<(&GenericMlp, bool)> {
        Some((&self.net, false))
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
    fn as_generic_mlp(&self) -> Option<(&GenericMlp, bool)> {
        Some((&self.net, true))
    }
}

// ── LoadedModel ──────────────────────────────────────────────────────────

use crate::model_registry::ModelRegistry;

/// A loaded model ready for evaluation. Can represent a DB-registered model,
/// a file-based model, or a built-in player (base/random).
pub struct LoadedModel {
    /// DB model ID, None if not in registry.
    pub id: Option<i64>,
    /// Short display label (e.g. "Base", "Random", "es_final (1106->64->64->32->1)").
    pub label: String,
    /// None = Random player (no evaluator needed).
    pub evaluator: Option<Box<dyn GameEvaluator + Sync + Send>>,
}

impl LoadedModel {
    /// Load from a spec string: "base", "random", or a file path (.gmlp/.nnue).
    pub fn from_spec(spec: &str) -> Self {
        let (eval, desc) = load_opponent(spec);
        let label = if spec == "base" || spec == "random" {
            spec.to_string()
        } else {
            // Use filename stem + arch from description
            let stem = std::path::Path::new(spec)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| spec.to_string());
            // Extract parenthesized part from desc if present
            if let Some(start) = desc.find('(') {
                format!("{} {}", stem, &desc[start..])
            } else {
                stem
            }
        };
        LoadedModel { id: None, label, evaluator: eval }
    }

    /// Load from a database model ID. Opens the registry, looks up the model,
    /// loads the file, and populates the struct.
    pub fn from_db_id(registry: &ModelRegistry, model_id: i64) -> Result<Self, String> {
        let record = registry
            .get_model(model_id)
            .map_err(|e| format!("DB error looking up model #{}: {}", model_id, e))?
            .ok_or_else(|| format!("Model #{} not found in registry", model_id))?;

        let (eval, _desc) = load_opponent(&record.file_path);
        let label = if let Some(ref desc) = record.description {
            desc.clone()
        } else {
            format!("{} ({})",
                std::path::Path::new(&record.file_path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| record.file_path.clone()),
                record.architecture)
        };

        Ok(LoadedModel {
            id: Some(model_id),
            label,
            evaluator: eval,
        })
    }

    /// Convert to a Player reference for match_runner.
    pub fn as_player(&self) -> crate::match_runner::Player<'_> {
        match &self.evaluator {
            Some(eval) => crate::match_runner::Player::Evaluator(eval.as_ref()),
            None => crate::match_runner::Player::Random,
        }
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
        path if path.ends_with(".json") => {
            let raw = load_lr_weights_raw(path).expect("Failed to load .json LR weights");
            let n = raw.len();
            match n {
                LR_NUM_FEATURES => {
                    let mut weights = [0.0f64; LR_NUM_FEATURES];
                    weights.copy_from_slice(&raw);
                    let lhw = LearnedHeuristicWeights { weights };
                    let desc = format!("LR-Guard ({} weights)", n);
                    (Some(Box::new(lhw)), desc)
                }
                NUM_COMBINED_FEATURES => {
                    let mut weights = [0.0f64; NUM_COMBINED_FEATURES];
                    weights.copy_from_slice(&raw);
                    let cw = CombinedWeights { weights };
                    let desc = format!("LR-Cheap ({} weights)", n);
                    (Some(Box::new(cw)), desc)
                }
                NUM_ALL_FEATURES => {
                    let mut weights = [0.0f64; NUM_ALL_FEATURES];
                    weights.copy_from_slice(&raw);
                    let aw = AllFeaturesWeights { weights };
                    let desc = format!("LR-All ({} weights)", n);
                    (Some(Box::new(aw)), desc)
                }
                _ => {
                    panic!(
                        "JSON weight file '{}' has {} weights. Expected {} (LR-Guard), {} (LR-Cheap), or {} (LR-All).",
                        path, n, LR_NUM_FEATURES, NUM_COMBINED_FEATURES, NUM_ALL_FEATURES
                    );
                }
            }
        }
        other => {
            panic!(
                "Unknown opponent '{}'. Use 'base', 'random', or a path ending in .gmlp / .nnue / .json",
                other
            );
        }
    }
}
