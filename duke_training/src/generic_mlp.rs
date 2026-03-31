//! Generic MLP network and evaluator wrappers, shared between es_train and elo_tournament.

use rand::rngs::StdRng;
use rand::Rng;

use duke_rust::game::state::GameState;

use crate::encoding::{
    active_board_features, bag_features, FeatureBuffer, BOARD_FEATURES, BAG_FEATURES,
    TOTAL_FEATURES,
};
use crate::game_setup::GameEvaluator;
use crate::learned_heuristic::{
    extract_combined_features, extract_features, NUM_COMBINED_FEATURES,
    NUM_FEATURES as LH_NUM_FEATURES,
};

// ── Generic MLP network (N hidden layers) ────────────────────────────────
// A self-contained input->H1->H2->...->Hn->1 network with ReLU hidden layers
// and sigmoid output. Operates entirely on Vec<f32>.

/// Maximum hidden layer size for stack allocation in forward passes.
pub const MAX_HIDDEN: usize = 1024;

// NUM_GUARD_ALL_FEATURES lives in loaded_model.rs; re-import for local use.
use crate::loaded_model::NUM_GUARD_ALL_FEATURES;

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
        debug_assert!(self.input_size >= TOTAL_FEATURES,
            "forward_sparse requires input_size >= {} (TOTAL_FEATURES), got {}",
            TOTAL_FEATURES, self.input_size);
        debug_assert!(!include_combined || self.input_size >= TOTAL_FEATURES + NUM_COMBINED_FEATURES,
            "forward_sparse with include_combined requires input_size >= {}, got {}",
            TOTAL_FEATURES + NUM_COMBINED_FEATURES, self.input_size);
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

    /// Forward pass for the 1171-input "all appended" mode:
    /// 1106 sparse board features + 41 cheap combined + 24 expensive guard features.
    ///
    /// This is like `forward_sparse(gs, true)` but additionally accumulates the
    /// 24 expensive features from `extract_features` at indices 1147..1171.
    pub fn forward_sparse_all(&self, gs: &GameState) -> f32 {
        debug_assert_eq!(self.input_size, ALL_APPENDED_INPUT_SIZE,
            "forward_sparse_all requires input_size == {} (ALL_APPENDED_INPUT_SIZE), got {}",
            ALL_APPENDED_INPUT_SIZE, self.input_size);
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

        // Combined features (dimensions 1106..1147)
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

        // Expensive guard features (dimensions 1147..1171)
        let expensive = extract_features(gs);
        for (i, &val) in expensive.iter().enumerate() {
            let fval = val as f32;
            if fval != 0.0 {
                let feat = TOTAL_FEATURES + NUM_COMBINED_FEATURES + i;
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

// ── Quantized MLP (int8 weights, f32 activations) ────────────────────────

/// A single quantized layer: i8 weights + f32 biases + scale factor.
struct QuantizedLayer {
    /// Row-major quantized weights: prev_size × cur_size, stored as i8.
    weights_i8: Vec<i8>,
    /// Bias vector (cur_size), kept as f32.
    biases: Vec<f32>,
    /// Scale factor: real_weight = weight_i8 * weight_scale.
    weight_scale: f32,
    /// Previous layer size (for indexing into weights).
    prev_size: usize,
    /// Current layer size.
    cur_size: usize,
}

/// Int8 quantized version of [`GenericMlp`].
///
/// Weights are quantized to i8 with a per-layer scale factor.
/// Activations remain f32 throughout. The forward pass computes
/// `sum += (w_i8 as f32) * activation` per element, then multiplies
/// by `weight_scale` once per output neuron. This reduces weight memory
/// by 4x and allows the compiler to auto-vectorize the i8->f32 cast +
/// multiply pattern.
///
/// The output layer is kept as f32 (tiny -- just `last_hidden + 1` params).
pub struct QuantizedGenericMlp {
    pub input_size: usize,
    pub hidden_layers: Vec<usize>,
    /// Quantized hidden layers (layers 1..N, not including L1 or output).
    hidden_quantized: Vec<QuantizedLayer>,
    /// L1 weights stay f32 for sparse accumulation compatibility.
    l1_weights_f32: Vec<f32>,
    /// L1 biases (f32).
    l1_biases: Vec<f32>,
    /// Output layer weights (f32) -- tiny, not worth quantizing.
    output_weights: Vec<f32>,
    /// Output layer bias (f32).
    output_bias: f32,
}

/// Quantize a single f32 weight value to i8 with the given scale.
#[inline]
fn quantize_weight(w: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        return 0;
    }
    (w / scale).round().max(-127.0).min(127.0) as i8
}

impl GenericMlp {
    /// Create a quantized version of this network.
    ///
    /// L1 (first hidden layer) weights are kept as f32 because they participate
    /// in sparse accumulation with the `L1Accumulator`. Hidden layers 2..N are
    /// quantized to i8 with per-layer scale factors. The output layer stays f32.
    pub fn quantize(&self) -> QuantizedGenericMlp {
        let w = &self.weights;
        let h1 = self.hidden_layers[0];

        // Extract L1 weights and biases (keep f32)
        let l1_weights_f32 = w[0..self.input_size * h1].to_vec();
        let l1_biases = w[self.input_size * h1..self.input_size * h1 + h1].to_vec();
        let mut off = self.input_size * h1 + h1;

        // Quantize hidden layers 2..N
        let mut hidden_quantized = Vec::new();
        for layer_idx in 1..self.hidden_layers.len() {
            let prev_size = self.hidden_layers[layer_idx - 1];
            let cur_size = self.hidden_layers[layer_idx];
            let n_weights = prev_size * cur_size;
            let lw = &w[off..off + n_weights];
            off += n_weights;
            let lb = &w[off..off + cur_size];
            off += cur_size;

            // Find max absolute weight for this layer
            let max_abs = lw.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
            let weight_scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

            // Quantize weights
            let weights_i8: Vec<i8> = lw.iter().map(|&v| quantize_weight(v, weight_scale)).collect();

            hidden_quantized.push(QuantizedLayer {
                weights_i8,
                biases: lb.to_vec(),
                weight_scale,
                prev_size,
                cur_size,
            });
        }

        // Extract output layer (keep f32)
        let last_h = *self.hidden_layers.last().unwrap();
        let output_weights = w[off..off + last_h].to_vec();
        off += last_h;
        let output_bias = w[off];

        QuantizedGenericMlp {
            input_size: self.input_size,
            hidden_layers: self.hidden_layers.clone(),
            hidden_quantized,
            l1_weights_f32,
            l1_biases,
            output_weights,
            output_bias,
        }
    }
}

impl QuantizedGenericMlp {
    /// Shared inner forward pass: propagate through quantized hidden layers 2..N
    /// and the f32 output sigmoid.
    ///
    /// `buf_a` must already contain the ReLU-activated first hidden layer output.
    /// `buf_b` is scratch space. Both are `[f32; MAX_HIDDEN]` ping-pong buffers.
    fn forward_inner(
        &self,
        buf_a: &mut [f32; MAX_HIDDEN],
        buf_b: &mut [f32; MAX_HIDDEN],
    ) -> f32 {
        let mut use_a = true;

        for layer in &self.hidden_quantized {
            let (src, dst) = if use_a {
                (&*buf_a as &[f32; MAX_HIDDEN], &mut *buf_b)
            } else {
                (&*buf_b as &[f32; MAX_HIDDEN], &mut *buf_a)
            };

            // Zero-initialize the accumulator (dot product in quantized units)
            for j in 0..layer.cur_size {
                dst[j] = 0.0;
            }

            // Matrix-vector multiply: dst += W_i8 * src (in quantized units)
            // Iterate row-major: for each input neuron i, scatter its contribution.
            let w_i8 = &layer.weights_i8;
            for i in 0..layer.prev_size {
                let w_row = &w_i8[i * layer.cur_size..(i + 1) * layer.cur_size];
                let s = src[i];
                if s != 0.0 {
                    for j in 0..layer.cur_size {
                        dst[j] += (w_row[j] as f32) * s;
                    }
                }
            }

            // Dequantize: multiply by weight_scale to get real units, add bias, ReLU
            let scale = layer.weight_scale;
            for j in 0..layer.cur_size {
                dst[j] = (dst[j] * scale + layer.biases[j]).max(0.0);
            }

            use_a = !use_a;
        }

        // Output layer: last_hidden -> 1, sigmoid
        let last_h = *self.hidden_layers.last().unwrap();
        let prev = if use_a { &*buf_a } else { &*buf_b };
        let mut logit = self.output_bias;
        for i in 0..last_h {
            logit += self.output_weights[i] * prev[i];
        }

        1.0 / (1.0 + (-logit).exp())
    }

    /// Complete the forward pass from a pre-computed L1 accumulator.
    ///
    /// Applies ReLU to the L1 values, then propagates through the quantized
    /// hidden layers and outputs a sigmoid score.
    pub fn forward_from_l1(&self, l1: &L1Accumulator) -> f32 {
        let h1 = l1.h1_size;
        let mut buf_a = [0.0f32; MAX_HIDDEN];
        for j in 0..h1 {
            buf_a[j] = l1.hidden[j].max(0.0); // ReLU
        }
        let mut buf_b = [0.0f32; MAX_HIDDEN];
        self.forward_inner(&mut buf_a, &mut buf_b)
    }

    /// Forward pass optimized for sparse NNUE-style inputs.
    ///
    /// L1 accumulation is identical to `GenericMlp::forward_sparse` (f32 sparse
    /// accumulation). Subsequent hidden layers use quantized i8 weights.
    pub fn forward_sparse(&self, gs: &GameState, include_combined: bool) -> f32 {
        let h1 = self.hidden_layers[0];

        // L1: sparse accumulation using f32 weights (same as GenericMlp)
        let l1_w = &self.l1_weights_f32;
        let mut buf_a = [0.0f32; MAX_HIDDEN];
        buf_a[..h1].copy_from_slice(&self.l1_biases);

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
        self.forward_inner(&mut buf_a, &mut buf_b)
    }

    /// Format the architecture as a string like "1106->64->64->32->1 (q)"
    pub fn arch_string(&self) -> String {
        let mut s = format!("{}", self.input_size);
        for &h in &self.hidden_layers {
            s.push_str(&format!("->{}", h));
        }
        s.push_str("->1 (q)");
        s
    }
}

/// Evaluator that wraps a QuantizedGenericMlp, dispatching by input_size.
pub struct QuantizedEvaluator {
    pub qnet: QuantizedGenericMlp,
}

impl GameEvaluator for QuantizedEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        match self.qnet.input_size {
            TOTAL_FEATURES => self.qnet.forward_sparse(gs, false),
            APPENDED_INPUT_SIZE => self.qnet.forward_sparse(gs, true),
            other => panic!(
                "QuantizedEvaluator: unsupported input_size {}. Expected {} or {}.",
                other, TOTAL_FEATURES, APPENDED_INPUT_SIZE,
            ),
        }
    }
    // as_generic_mlp returns None: the incremental L1 accumulator path
    // calls forward_inner on the f32 GenericMlp, which would bypass
    // quantization. Instead we fall back to the evaluate() path above.
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
/// This avoids redundant L1 recomputation across candidates -- only the
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
    /// Build an L1 accumulator from raw pre-computed hidden values.
    ///
    /// `hidden` should contain the raw L1 values (bias + weighted sum) before ReLU.
    /// `h1_size` is the number of active hidden units.
    pub fn from_raw(hidden: [f32; MAX_HIDDEN], h1_size: usize) -> Self {
        Self { hidden, h1_size }
    }

    /// Build an L1 accumulator from a game state (full recompute).
    ///
    /// Performs the same sparse accumulation as `forward_sparse`, but stops
    /// before applying ReLU, storing the raw weighted sums.
    pub fn from_state(net: &GenericMlp, gs: &GameState, include_combined: bool) -> Self {
        debug_assert!(net.input_size >= TOTAL_FEATURES,
            "L1Accumulator::from_state requires net.input_size >= {} (TOTAL_FEATURES), got {}",
            TOTAL_FEATURES, net.input_size);
        debug_assert!(!include_combined || net.input_size >= TOTAL_FEATURES + NUM_COMBINED_FEATURES,
            "L1Accumulator::from_state with include_combined requires net.input_size >= {}, got {}",
            TOTAL_FEATURES + NUM_COMBINED_FEATURES, net.input_size);
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
        // Use bitmaps for O(1) set membership instead of O(n) contains() per feature.
        let mut old_set = [0u64; (BOARD_FEATURES + 63) / 64];
        for &feat in old_board.as_slice() {
            old_set[feat / 64] |= 1u64 << (feat % 64);
        }
        let mut new_set = [0u64; (BOARD_FEATURES + 63) / 64];
        for &feat in new_board.as_slice() {
            new_set[feat / 64] |= 1u64 << (feat % 64);
        }
        // Remove old board features not in new set
        for &feat in old_board.as_slice() {
            if new_set[feat / 64] & (1u64 << (feat % 64)) == 0 {
                let col = &l1_w[feat * h1..(feat + 1) * h1];
                for j in 0..h1 {
                    self.hidden[j] -= col[j];
                }
            }
        }
        // Add new board features not in old set
        for &feat in new_board.as_slice() {
            if old_set[feat / 64] & (1u64 << (feat % 64)) == 0 {
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

    /// Complete the forward pass using a quantized network.
    ///
    /// Same as [`forward`], but uses the quantized hidden layers from
    /// `QuantizedGenericMlp` instead of the f32 `GenericMlp`.
    pub fn forward_quantized(&self, qnet: &QuantizedGenericMlp) -> f32 {
        qnet.forward_from_l1(self)
    }
}

/// Maximum candidate batch size for batched forward passes.
/// Greedy move search typically has 20-40 candidates.
pub const MAX_BATCH: usize = 128;

impl GenericMlp {
    /// Batched forward pass from multiple L1 accumulators.
    ///
    /// Takes a slice of pre-ReLU L1 hidden values (one per candidate),
    /// applies ReLU, then propagates through all remaining hidden layers
    /// using sgemm (one call per layer for the entire batch), and returns
    /// sigmoid outputs for each candidate.
    ///
    /// Much faster than calling `L1Accumulator::forward` N times because
    /// sgemm with m=N exploits cache tiling and SIMD across the batch.
    pub fn batched_forward_from_l1(
        &self,
        accumulators: &[L1Accumulator],
        outputs: &mut [f32],
    ) {
        use matrixmultiply::sgemm;

        let n = accumulators.len();
        assert!(n <= MAX_BATCH);
        assert!(outputs.len() >= n);
        let h1 = self.hidden_layers[0];
        let off_start = self.input_size * h1 + h1;

        // Build the batch activation matrix: [n × h1], row-major.
        // Apply ReLU to each accumulator's hidden values.
        let mut act_a = vec![0.0f32; n * MAX_HIDDEN];
        for (i, acc) in accumulators.iter().enumerate() {
            let row = &mut act_a[i * MAX_HIDDEN..(i * MAX_HIDDEN + h1)];
            for j in 0..h1 {
                row[j] = acc.hidden[j].max(0.0);
            }
        }
        let mut act_b = vec![0.0f32; n * MAX_HIDDEN];

        let mut use_a = true;
        let mut off = off_start;

        for layer_idx in 1..self.hidden_layers.len() {
            let prev_size = self.hidden_layers[layer_idx - 1];
            let cur_size = self.hidden_layers[layer_idx];
            let lw = &self.weights[off..off + prev_size * cur_size];
            off += prev_size * cur_size;
            let lb = &self.weights[off..off + cur_size];
            off += cur_size;

            let (src, dst) = if use_a {
                (&act_a, &mut act_b)
            } else {
                (&act_b, &mut act_a)
            };

            // Initialize dst rows with bias
            for i in 0..n {
                dst[i * MAX_HIDDEN..(i * MAX_HIDDEN + cur_size)].copy_from_slice(lb);
            }

            // dst[n×cur] += src[n×prev] × W[prev×cur]
            // src is row-major with stride MAX_HIDDEN, W is row-major with stride cur_size
            unsafe {
                sgemm(
                    n, prev_size, cur_size,
                    1.0,
                    src.as_ptr(), MAX_HIDDEN as isize, 1,
                    lw.as_ptr(), cur_size as isize, 1,
                    1.0,
                    dst.as_mut_ptr(), MAX_HIDDEN as isize, 1,
                );
            }

            // ReLU
            for i in 0..n {
                let row = &mut dst[i * MAX_HIDDEN..(i * MAX_HIDDEN + cur_size)];
                for j in 0..cur_size {
                    row[j] = row[j].max(0.0);
                }
            }
            use_a = !use_a;
        }

        // Output layer: dot product per candidate
        let last_h = *self.hidden_layers.last().unwrap();
        let out_w = &self.weights[off..off + last_h];
        off += last_h;
        let out_b = self.weights[off];

        let prev = if use_a { &act_a } else { &act_b };
        for i in 0..n {
            let row = &prev[i * MAX_HIDDEN..(i * MAX_HIDDEN + last_h)];
            let mut logit = out_b;
            for j in 0..last_h {
                logit += out_w[j] * row[j];
            }
            outputs[i] = 1.0 / (1.0 + (-logit).exp());
        }
    }
}

/// Total input size for the "appended" mode:
/// NNUE features + combined features (e.g. 1106 + 41 = 1147).
pub const APPENDED_INPUT_SIZE: usize = TOTAL_FEATURES + NUM_COMBINED_FEATURES;

/// Total input size for the "all appended" mode:
/// NNUE features + combined + expensive guard features (e.g. 1106 + 41 + 24 = 1171).
pub const ALL_APPENDED_INPUT_SIZE: usize = TOTAL_FEATURES + NUM_COMBINED_FEATURES + LH_NUM_FEATURES;

/// Unified evaluator that wraps a GenericMlp, dispatching by input_size.
///
/// Replaces the former CombinedNetEvaluator (41), GuardFeatureEvaluator (65),
/// GenericNnueEvaluator (1106), GenericAppendedEvaluator (1147), and
/// AllAppendedEvaluator (1171) with a single struct.
pub struct GenericEvaluator {
    pub net: GenericMlp,
}

impl GameEvaluator for GenericEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        match self.net.input_size {
            NUM_COMBINED_FEATURES => {
                let features = extract_combined_features(gs);
                self.net.forward_f64(&features)
            }
            NUM_GUARD_ALL_FEATURES => {
                let expensive = extract_features(gs);
                let combined = extract_combined_features(gs);
                let mut features = [0.0f64; NUM_GUARD_ALL_FEATURES];
                features[..LH_NUM_FEATURES].copy_from_slice(&expensive);
                features[LH_NUM_FEATURES..].copy_from_slice(&combined);
                self.net.forward_f64(&features)
            }
            TOTAL_FEATURES => self.net.forward_sparse(gs, false),
            APPENDED_INPUT_SIZE => self.net.forward_sparse(gs, true),
            ALL_APPENDED_INPUT_SIZE => self.net.forward_sparse_all(gs),
            other => panic!(
                "GenericEvaluator: unknown input_size {}. Expected {}, {}, {}, {}, or {}.",
                other, NUM_COMBINED_FEATURES, NUM_GUARD_ALL_FEATURES,
                TOTAL_FEATURES, APPENDED_INPUT_SIZE, ALL_APPENDED_INPUT_SIZE,
            ),
        }
    }

    fn as_generic_mlp(&self) -> Option<(&GenericMlp, bool)> {
        match self.net.input_size {
            TOTAL_FEATURES => Some((&self.net, false)),
            APPENDED_INPUT_SIZE => Some((&self.net, true)),
            _ => None, // dense and all-appended don't support incremental
        }
    }
}

