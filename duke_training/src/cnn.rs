//! Manual CNN implementation (forward + backward) for supervised training.
//!
//! Pure Rust — no burn framework. Supports three kernel types:
//! - **Box**: standard 3x3 convolution with pad=1
//! - **Diamond**: 5x5 with manhattan-2 mask (13 active positions), pad=2
//! - **Cross**: 3x3 + 5x1 + 1x5 summed, with appropriate padding
//!
//! Network: Conv layers → flatten → concat bag features → FC layers → sigmoid output.

use rand::rngs::StdRng;
use rand::Rng;

use duke_rust::game::state::GameState;

use crate::encoding::{active_board_features, bag_features};
use crate::game_setup::GameEvaluator;

// ── Kernel type ───────────────────────────────────────────────────────────

/// Kernel type for CNN conv layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    /// Standard 3x3 convolution with pad=1.
    Box,
    /// 5x5 convolution with diamond (manhattan-2) mask zeroing corners, pad=2.
    Diamond,
    /// Three parallel convolutions summed: 3x3 (pad=1) + 1x5 (pad=0,2) + 5x1 (pad=2,0).
    Cross,
}

/// Build the diamond mask for a 5x5 kernel: manhattan distance <= 2 from center.
///
/// ```text
/// 0 0 1 0 0
/// 0 1 1 1 0
/// 1 1 1 1 1
/// 0 1 1 1 0
/// 0 0 1 0 0
/// ```
pub fn diamond_mask_5x5() -> [bool; 25] {
    let mut mask = [false; 25];
    for dy in 0..5i32 {
        for dx in 0..5i32 {
            let dist = (dy - 2).unsigned_abs() + (dx - 2).unsigned_abs();
            if dist <= 2 {
                mask[(dy * 5 + dx) as usize] = true;
            }
        }
    }
    mask
}

/// Precomputed list of (ky, kx, flat_index) for the 13 active diamond positions.
/// Avoids checking the mask in inner loops.
pub(crate) const DIAMOND_OFFSETS: [(i32, i32, usize); 13] = [
    (0, 2, 2),
    (1, 1, 6),
    (1, 2, 7),
    (1, 3, 8),
    (2, 0, 10),
    (2, 1, 11),
    (2, 2, 12),
    (2, 3, 13),
    (2, 4, 14),
    (3, 1, 16),
    (3, 2, 17),
    (3, 3, 18),
    (4, 2, 22),
];

// ── CnnModel ──────────────────────────────────────────────────────────────

/// Manual CNN value network. All parameters stored in a single flat vector.
pub struct CnnModel {
    pub kernel_type: KernelType,
    /// Channel count for each conv layer, e.g. [64, 64, 32].
    pub conv_channels: Vec<usize>,
    /// Hidden FC layer sizes (not including the output), e.g. [128].
    pub fc_sizes: Vec<usize>,
    /// Flat weight vector containing all conv + FC parameters.
    pub weights: Vec<f32>,
    /// Number of input channels (board planes). Typically 30.
    pub input_channels: usize,
    /// Spatial board dimension. Typically 6.
    pub board_size: usize,
    /// Number of bag features. Typically 26.
    pub bag_features: usize,
}

/// Cached layout information for a CnnModel, computed once and reused
/// to avoid per-call Vec allocations in forward/backward passes.
struct CnnLayout {
    /// Conv layer weight offsets (len = conv_channels.len() + 1, last is FC start).
    conv_offsets: Vec<usize>,
    /// FC layer weight offsets (len = fc_sizes.len() + 1, last is output layer).
    fc_offsets: Vec<usize>,
    /// Kernel weights per (out_ch, in_ch) pair.
    kwpp: usize,
}

/// Forward pass result containing intermediate values needed for backpropagation.
pub struct CnnForwardResult {
    /// Pre-ReLU activations per conv layer: [layer][out_ch * board_size * board_size].
    pub conv_pre_relu: Vec<Vec<f32>>,
    /// Post-ReLU activations per conv layer: [layer][out_ch * board_size * board_size].
    pub conv_post_relu: Vec<Vec<f32>>,
    /// Pre-ReLU activations per FC hidden layer: [layer][neurons].
    pub fc_pre_relu: Vec<Vec<f32>>,
    /// Post-ReLU activations per FC hidden layer: [layer][neurons].
    pub fc_post_relu: Vec<Vec<f32>>,
    /// Final sigmoid output.
    pub output: f32,
}

/// Pre-allocated scratch buffers for CNN forward/backward passes.
///
/// Create once via `CnnModel::create_scratch()` and reuse across all
/// positions in a training loop to eliminate per-position heap allocations.
pub struct CnnScratch {
    // ── Forward (inference) ping-pong buffers ──
    /// Conv buffer A (max_channels * spatial).
    conv_buf_a: Vec<f32>,
    /// Conv buffer B (max_channels * spatial).
    conv_buf_b: Vec<f32>,
    /// FC activation buffer A.
    fc_buf_a: Vec<f32>,
    /// FC activation buffer B.
    fc_buf_b: Vec<f32>,

    // ── Forward with intermediates ──
    /// Pre-ReLU activations per conv layer.
    conv_pre_relu: Vec<Vec<f32>>,
    /// Post-ReLU activations per conv layer.
    conv_post_relu: Vec<Vec<f32>>,
    /// FC input (flattened conv + bag).
    fc_input: Vec<f32>,
    /// Pre-ReLU activations per FC hidden layer.
    fc_pre_relu: Vec<Vec<f32>>,
    /// Post-ReLU activations per FC hidden layer.
    fc_post_relu: Vec<Vec<f32>>,
    /// Prev activation buffer for FC forward.
    fc_prev_act: Vec<f32>,

    // ── im2col buffer for conv layers ──
    /// im2col buffer: [max_channels * max_kernel_area, spatial] for im2col+sgemm.
    im2col_buf: Vec<f32>,

    // ── Backward buffers ──
    /// d_next / d_prev for FC backward.
    bk_d_fc_a: Vec<f32>,
    /// d_h for FC backward.
    bk_d_fc_b: Vec<f32>,
    /// d_post for conv backward.
    bk_d_conv: Vec<f32>,
    /// d_input for conv backward (propagated to previous layer).
    bk_d_input: Vec<f32>,
    /// fc_input reconstruction for backward.
    bk_fc_input: Vec<f32>,
    /// Cached layout (avoids per-call Vec allocation of offsets).
    layout: CnnLayout,
}

// ── Weight layout helpers ─────────────────────────────────────────────────

impl CnnLayout {
    /// Build cached layout from model parameters.
    fn new(model: &CnnModel) -> Self {
        let kwpp = CnnModel::kernel_weights_per_pair(model.kernel_type);

        // Conv offsets
        let mut conv_offsets = Vec::with_capacity(model.conv_channels.len() + 1);
        let mut off = 0;
        let mut in_ch = model.input_channels;
        for &out_ch in &model.conv_channels {
            conv_offsets.push(off);
            off += CnnModel::conv_layer_params(model.kernel_type, in_ch, out_ch);
            in_ch = out_ch;
        }
        conv_offsets.push(off); // FC section start

        // FC offsets
        let fc_start = off;
        let mut fc_offsets = Vec::with_capacity(model.fc_sizes.len() + 2);
        let mut fc_off = fc_start;
        let mut prev = model.fc_input_size();
        for &h in &model.fc_sizes {
            fc_offsets.push(fc_off);
            fc_off += prev * h + h;
            prev = h;
        }
        fc_offsets.push(fc_off); // output layer

        Self { conv_offsets, fc_offsets, kwpp }
    }
}

impl CnnModel {
    /// Kernel footprint (number of f32 weights per (out_ch, in_ch) pair) for a single conv.
    #[inline]
    fn kernel_weights_per_pair(kernel: KernelType) -> usize {
        match kernel {
            KernelType::Box => 3 * 3,
            KernelType::Diamond => 5 * 5,
            KernelType::Cross => 3 * 3 + 1 * 5 + 5 * 1, // box + h + v
        }
    }

    /// Total weight count for one conv layer (excluding bias for Cross sub-convs — bias
    /// is shared as a single [out_ch] bias vector per layer).
    fn conv_layer_params(kernel: KernelType, in_ch: usize, out_ch: usize) -> usize {
        let weights = out_ch * in_ch * Self::kernel_weights_per_pair(kernel);
        let bias = out_ch;
        weights + bias
    }

    /// Compute total parameter count for the given architecture.
    pub fn param_count(
        kernel_type: KernelType,
        input_channels: usize,
        conv_channels: &[usize],
        fc_sizes: &[usize],
        board_size: usize,
        bag_features: usize,
    ) -> usize {
        assert!(!conv_channels.is_empty(), "Need at least one conv layer");
        let mut count = 0;
        let mut in_ch = input_channels;
        for &out_ch in conv_channels {
            count += Self::conv_layer_params(kernel_type, in_ch, out_ch);
            in_ch = out_ch;
        }
        // FC input: flattened conv output + bag features
        let fc_input = conv_channels.last().unwrap() * board_size * board_size + bag_features;
        let mut prev = fc_input;
        for &h in fc_sizes {
            count += prev * h + h; // weight + bias
            prev = h;
        }
        count += prev + 1; // output layer: weight + bias
        count
    }

    /// Compute the size of the FC input (flattened conv + bag).
    #[inline]
    pub fn fc_input_size(&self) -> usize {
        self.conv_channels.last().unwrap() * self.board_size * self.board_size + self.bag_features
    }

    /// Create a `CnnScratch` with all buffers pre-allocated for this model's architecture.
    /// Call once and reuse across the entire training loop.
    pub fn create_scratch(&self) -> CnnScratch {
        let spatial = self.board_size * self.board_size;
        let max_ch = *self.conv_channels.iter().max().unwrap();
        let max_conv_buf = max_ch * spatial;

        // FC sizes
        let fc_input_size = self.fc_input_size();
        let max_fc = self.fc_sizes.iter().copied().max().unwrap_or(0).max(fc_input_size);

        // Per-layer conv buffers
        let conv_pre_relu: Vec<Vec<f32>> = self.conv_channels.iter()
            .map(|&ch| vec![0.0f32; ch * spatial])
            .collect();
        let conv_post_relu: Vec<Vec<f32>> = self.conv_channels.iter()
            .map(|&ch| vec![0.0f32; ch * spatial])
            .collect();

        // Per-layer FC buffers
        let fc_pre_relu: Vec<Vec<f32>> = self.fc_sizes.iter()
            .map(|&h| vec![0.0f32; h])
            .collect();
        let fc_post_relu: Vec<Vec<f32>> = self.fc_sizes.iter()
            .map(|&h| vec![0.0f32; h])
            .collect();

        // For backward: max of all FC layer sizes and fc_input_size
        let bk_max_fc = self.fc_sizes.iter().copied().max().unwrap_or(0).max(fc_input_size);

        // im2col buffer: max_channels * max_kernel_area rows x spatial columns
        // Box: 9, Diamond: 25, Cross: max sub-kernel is 9 (3x3) but also needs 5 (1x5, 5x1)
        let max_kernel_area = match self.kernel_type {
            KernelType::Box => 9,
            KernelType::Diamond => 25,
            KernelType::Cross => 9, // largest sub-kernel is 3x3=9; sub-kernels run sequentially
        };
        let im2col_size = max_ch * max_kernel_area * spatial;

        CnnScratch {
            conv_buf_a: vec![0.0f32; max_conv_buf],
            conv_buf_b: vec![0.0f32; max_conv_buf],
            fc_buf_a: vec![0.0f32; max_fc],
            fc_buf_b: vec![0.0f32; max_fc],
            im2col_buf: vec![0.0f32; im2col_size],
            conv_pre_relu,
            conv_post_relu,
            fc_input: vec![0.0f32; fc_input_size],
            fc_pre_relu,
            fc_post_relu,
            fc_prev_act: vec![0.0f32; max_fc],
            bk_d_fc_a: vec![0.0f32; bk_max_fc],
            bk_d_fc_b: vec![0.0f32; bk_max_fc],
            bk_d_conv: vec![0.0f32; max_conv_buf],
            bk_d_input: vec![0.0f32; max_conv_buf],
            bk_fc_input: vec![0.0f32; fc_input_size],
            layout: CnnLayout::new(self),
        }
    }

    /// Create a randomly initialized CNN model (Kaiming uniform).
    pub fn random(
        kernel_type: KernelType,
        input_channels: usize,
        conv_channels: Vec<usize>,
        fc_sizes: Vec<usize>,
        board_size: usize,
        bag_features: usize,
        rng: &mut StdRng,
    ) -> Self {
        let n = Self::param_count(
            kernel_type,
            input_channels,
            &conv_channels,
            &fc_sizes,
            board_size,
            bag_features,
        );
        let mut weights = Vec::with_capacity(n);

        // Conv layers
        let mut in_ch = input_channels;
        for &out_ch in &conv_channels {
            let fan_in = in_ch * Self::kernel_weights_per_pair(kernel_type);
            let scale = (6.0 / fan_in as f64).sqrt() as f32;
            let num_w = out_ch * in_ch * Self::kernel_weights_per_pair(kernel_type);
            for _ in 0..num_w {
                weights.push(rng.gen::<f32>() * 2.0 * scale - scale);
            }
            // Bias = 0
            for _ in 0..out_ch {
                weights.push(0.0);
            }
            in_ch = out_ch;
        }

        // Apply diamond mask to initial weights if needed
        if kernel_type == KernelType::Diamond {
            let mask = diamond_mask_5x5();
            let mut offset = 0;
            let mut ic = input_channels;
            for &oc in &conv_channels {
                let n_w = oc * ic * 25;
                for i in 0..n_w {
                    let kpos = i % 25;
                    if !mask[kpos] {
                        weights[offset + i] = 0.0;
                    }
                }
                offset += n_w + oc; // weights + bias
                ic = oc;
            }
        }

        // FC layers
        let fc_input = conv_channels.last().unwrap() * board_size * board_size + bag_features;
        let mut prev = fc_input;
        for &h in &fc_sizes {
            let scale = (6.0 / prev as f64).sqrt() as f32;
            for _ in 0..(prev * h) {
                weights.push(rng.gen::<f32>() * 2.0 * scale - scale);
            }
            for _ in 0..h {
                weights.push(0.0);
            }
            prev = h;
        }
        // Output layer
        let scale = (6.0 / prev as f64).sqrt() as f32;
        for _ in 0..prev {
            weights.push(rng.gen::<f32>() * 2.0 * scale - scale);
        }
        weights.push(0.0);

        assert_eq!(
            weights.len(),
            n,
            "weight vector size mismatch: expected {}, got {}",
            n,
            weights.len()
        );

        CnnModel {
            kernel_type,
            conv_channels,
            fc_sizes,
            weights,
            input_channels,
            board_size,
            bag_features,
        }
    }

    /// Architecture description string.
    pub fn arch_string(&self) -> String {
        let kernel_str = match self.kernel_type {
            KernelType::Box => "box",
            KernelType::Diamond => "diamond",
            KernelType::Cross => "cross",
        };
        let conv_str: Vec<String> = self.conv_channels.iter().map(|c| c.to_string()).collect();
        let fc_str: Vec<String> = self.fc_sizes.iter().map(|c| c.to_string()).collect();
        format!(
            "CNN-{} conv=[{}] fc=[{}]->1",
            kernel_str,
            conv_str.join(","),
            fc_str.join(","),
        )
    }

    // ── Weight offset computation ─────────────────────────────────────────

    /// Compute the byte offset into the weight vector for each conv layer's start.
    /// Returns offsets for conv layers and the start of the FC section.
    fn conv_layer_offsets(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.conv_channels.len() + 1);
        let mut off = 0;
        let mut in_ch = self.input_channels;
        for &out_ch in &self.conv_channels {
            offsets.push(off);
            off += Self::conv_layer_params(self.kernel_type, in_ch, out_ch);
            in_ch = out_ch;
        }
        offsets.push(off); // FC section start
        offsets
    }

    /// Compute the start offset of each FC layer within the weight vector.
    fn fc_layer_offsets(&self) -> Vec<usize> {
        let conv_offsets = self.conv_layer_offsets();
        let fc_start = *conv_offsets.last().unwrap();
        let mut offsets = Vec::with_capacity(self.fc_sizes.len() + 2);
        let mut off = fc_start;
        let mut prev = self.fc_input_size();
        for &h in &self.fc_sizes {
            offsets.push(off);
            off += prev * h + h;
            prev = h;
        }
        offsets.push(off); // output layer
        offsets
    }

    // ── Forward pass (inference only) ─────────────────────────────────────

    /// Fast forward pass for inference. Uses sparse first conv layer.
    /// `active_board` contains flat indices into the 30*6*6 = 1080 board features
    /// (each index = plane * 36 + y * 6 + x).
    /// `bag` contains the 26 bag features.
    pub fn forward_sparse(&self, active_board: &[usize], bag: &[f32]) -> f32 {
        let bs = self.board_size;
        let spatial = bs * bs;
        let conv_offsets = self.conv_layer_offsets();

        // ── Conv layer 0: sparse input ──
        let out_ch0 = self.conv_channels[0];
        let off0 = conv_offsets[0];
        let kwpp = Self::kernel_weights_per_pair(self.kernel_type);
        let w0_size = out_ch0 * self.input_channels * kwpp;
        let bias0 = &self.weights[off0 + w0_size..off0 + w0_size + out_ch0];

        // Initialize output with bias (broadcast over spatial)
        let mut conv_out = vec![0.0f32; out_ch0 * spatial];
        for oc in 0..out_ch0 {
            let b = bias0[oc];
            for s in 0..spatial {
                conv_out[oc * spatial + s] = b;
            }
        }

        // Sparse accumulation: for each active feature, scatter contributions
        self.sparse_conv_accumulate(
            &self.weights[off0..off0 + w0_size],
            active_board,
            self.input_channels,
            out_ch0,
            &mut conv_out,
        );

        // ReLU
        for v in conv_out.iter_mut() {
            *v = v.max(0.0);
        }

        // ── Conv layers 1+ : dense ──
        for layer_idx in 1..self.conv_channels.len() {
            let in_ch = self.conv_channels[layer_idx - 1];
            let out_ch = self.conv_channels[layer_idx];
            let off = conv_offsets[layer_idx];
            let w_size = out_ch * in_ch * kwpp;
            let bias = &self.weights[off + w_size..off + w_size + out_ch];

            let prev = conv_out;
            conv_out = vec![0.0f32; out_ch * spatial];
            // Initialize with bias
            for oc in 0..out_ch {
                let b = bias[oc];
                for s in 0..spatial {
                    conv_out[oc * spatial + s] = b;
                }
            }
            self.dense_conv_accumulate(
                &self.weights[off..off + w_size],
                &prev,
                in_ch,
                out_ch,
                &mut conv_out,
            );
            // ReLU
            for v in conv_out.iter_mut() {
                *v = v.max(0.0);
            }
        }

        // ── Flatten + concat bag ──
        let fc_input_size = self.fc_input_size();
        let mut fc_input = Vec::with_capacity(fc_input_size);
        fc_input.extend_from_slice(&conv_out);
        fc_input.extend_from_slice(bag);
        debug_assert_eq!(fc_input.len(), fc_input_size);

        // ── FC layers ──
        let fc_offsets = self.fc_layer_offsets();
        let mut prev_act = fc_input;

        for (i, &h) in self.fc_sizes.iter().enumerate() {
            let off = fc_offsets[i];
            let prev_size = prev_act.len();
            let lw = &self.weights[off..off + prev_size * h];
            let lb = &self.weights[off + prev_size * h..off + prev_size * h + h];

            let mut cur = vec![0.0f32; h];
            cur.copy_from_slice(lb);
            for j in 0..prev_size {
                let w_row = &lw[j * h..(j + 1) * h];
                let s = prev_act[j];
                if s != 0.0 {
                    for k in 0..h {
                        cur[k] += w_row[k] * s;
                    }
                }
            }
            // ReLU
            for v in cur.iter_mut() {
                *v = v.max(0.0);
            }
            prev_act = cur;
        }

        // Output layer
        let out_off = *fc_offsets.last().unwrap();
        let last_h = prev_act.len();
        let out_w = &self.weights[out_off..out_off + last_h];
        let out_b = self.weights[out_off + last_h];

        let mut logit = out_b;
        for j in 0..last_h {
            logit += out_w[j] * prev_act[j];
        }

        sigmoid(logit)
    }

    /// Fast forward pass using pre-allocated scratch buffers (zero heap allocation).
    /// Uses ping-pong conv_buf_a / conv_buf_b for conv layers.
    pub fn forward_sparse_scratch(&self, active_board: &[usize], bag: &[f32], scratch: &mut CnnScratch) -> f32 {
        let bs = self.board_size;
        let spatial = bs * bs;
        let conv_offsets = &scratch.layout.conv_offsets;
        let kwpp = scratch.layout.kwpp;

        // Ping-pong: start writing to buf_a
        let (mut cur_buf, mut prev_buf) = (true, false); // true = buf_a, false = buf_b

        // ── Conv layer 0: sparse input ──
        let out_ch0 = self.conv_channels[0];
        let off0 = conv_offsets[0];
        let w0_size = out_ch0 * self.input_channels * kwpp;
        let bias0 = &self.weights[off0 + w0_size..off0 + w0_size + out_ch0];
        let size0 = out_ch0 * spatial;

        {
            let conv_out = if cur_buf { &mut scratch.conv_buf_a } else { &mut scratch.conv_buf_b };
            // Zero and init with bias
            for v in conv_out[..size0].iter_mut() { *v = 0.0; }
            for oc in 0..out_ch0 {
                let b = bias0[oc];
                for s in 0..spatial {
                    conv_out[oc * spatial + s] = b;
                }
            }
            self.sparse_conv_accumulate(
                &self.weights[off0..off0 + w0_size],
                active_board,
                self.input_channels,
                out_ch0,
                &mut conv_out[..size0],
            );
            // ReLU
            for v in conv_out[..size0].iter_mut() { *v = v.max(0.0); }
        }

        // ── Conv layers 1+ : dense ──
        for layer_idx in 1..self.conv_channels.len() {
            let in_ch = self.conv_channels[layer_idx - 1];
            let out_ch = self.conv_channels[layer_idx];
            let off = conv_offsets[layer_idx];
            let w_size = out_ch * in_ch * kwpp;
            let bias = &self.weights[off + w_size..off + w_size + out_ch];
            let in_size = in_ch * spatial;
            let out_size = out_ch * spatial;

            // Swap: previous output is in cur_buf, write new output to prev_buf
            std::mem::swap(&mut cur_buf, &mut prev_buf);

            // We need to split borrow: read from one, write to other
            // Use unsafe pointer trick to avoid double borrow
            let (prev_slice, next_slice) = if cur_buf {
                let (a, b) = (&scratch.conv_buf_b as &Vec<f32>, &mut scratch.conv_buf_a);
                (&a[..in_size], &mut b[..out_size])
            } else {
                let (a, b) = (&scratch.conv_buf_a as &Vec<f32>, &mut scratch.conv_buf_b);
                (&a[..in_size], &mut b[..out_size])
            };

            // Zero and init with bias
            for v in next_slice.iter_mut() { *v = 0.0; }
            for oc in 0..out_ch {
                let b = bias[oc];
                for s in 0..spatial {
                    next_slice[oc * spatial + s] = b;
                }
            }
            match self.kernel_type {
                KernelType::Box => {
                    Self::dense_conv_im2col_forward(
                        &self.weights[off..off + w_size],
                        prev_slice, in_ch, out_ch, bs,
                        3, 3, 1, 1,
                        &mut scratch.im2col_buf, next_slice,
                    );
                }
                KernelType::Diamond => {
                    Self::dense_conv_im2col_forward(
                        &self.weights[off..off + w_size],
                        prev_slice, in_ch, out_ch, bs,
                        5, 5, 2, 2,
                        &mut scratch.im2col_buf, next_slice,
                    );
                }
                KernelType::Cross => {
                    Self::dense_conv_im2col_forward_cross(
                        &self.weights[off..off + w_size],
                        prev_slice, in_ch, out_ch, bs,
                        &mut scratch.im2col_buf, next_slice,
                    );
                }
            }
            // ReLU
            for v in next_slice.iter_mut() { *v = v.max(0.0); }
        }

        // ── Flatten + concat bag into fc_buf_a ──
        let fc_input_size = self.fc_input_size();
        let last_ch = *self.conv_channels.last().unwrap();
        let conv_flat_size = last_ch * spatial;
        {
            let conv_out = if cur_buf { &scratch.conv_buf_a } else { &scratch.conv_buf_b };
            scratch.fc_buf_a[..conv_flat_size].copy_from_slice(&conv_out[..conv_flat_size]);
        }
        scratch.fc_buf_a[conv_flat_size..fc_input_size].copy_from_slice(bag);

        // ── FC layers: ping-pong fc_buf_a / fc_buf_b ──
        let fc_offsets = &scratch.layout.fc_offsets;
        let mut fc_cur = true; // true = fc_buf_a, false = fc_buf_b
        let mut prev_size = fc_input_size;

        for (i, &h) in self.fc_sizes.iter().enumerate() {
            let off = fc_offsets[i];
            let lw = &self.weights[off..off + prev_size * h];
            let lb = &self.weights[off + prev_size * h..off + prev_size * h + h];

            // Read from fc_cur, write to !fc_cur
            let (prev_act, cur) = if fc_cur {
                (&scratch.fc_buf_a[..prev_size], &mut scratch.fc_buf_b[..h])
            } else {
                (&scratch.fc_buf_b[..prev_size], &mut scratch.fc_buf_a[..h])
            };

            cur.copy_from_slice(lb);
            for j in 0..prev_size {
                let w_row = &lw[j * h..(j + 1) * h];
                let s = prev_act[j];
                if s != 0.0 {
                    for k in 0..h {
                        cur[k] += w_row[k] * s;
                    }
                }
            }
            // ReLU
            for v in cur.iter_mut() { *v = v.max(0.0); }

            fc_cur = !fc_cur;
            prev_size = h;
        }

        // Output layer
        let out_off = *fc_offsets.last().unwrap();
        let last_h = prev_size;
        let out_w = &self.weights[out_off..out_off + last_h];
        let out_b = self.weights[out_off + last_h];
        let prev_act = if fc_cur { &scratch.fc_buf_a[..last_h] } else { &scratch.fc_buf_b[..last_h] };

        let mut logit = out_b;
        for j in 0..last_h {
            logit += out_w[j] * prev_act[j];
        }

        sigmoid(logit)
    }

    // ── Forward pass with intermediates (for backprop) ────────────────────

    /// Forward pass saving all intermediate activations for backpropagation.
    pub fn forward_with_intermediates(
        &self,
        active_board: &[usize],
        bag: &[f32],
    ) -> CnnForwardResult {
        let bs = self.board_size;
        let spatial = bs * bs;
        let conv_offsets = self.conv_layer_offsets();
        let kwpp = Self::kernel_weights_per_pair(self.kernel_type);

        let mut conv_pre_relu = Vec::with_capacity(self.conv_channels.len());
        let mut conv_post_relu = Vec::with_capacity(self.conv_channels.len());

        // ── Conv layer 0: sparse input ──
        {
            let out_ch = self.conv_channels[0];
            let off = conv_offsets[0];
            let w_size = out_ch * self.input_channels * kwpp;
            let bias = &self.weights[off + w_size..off + w_size + out_ch];

            let mut pre = vec![0.0f32; out_ch * spatial];
            for oc in 0..out_ch {
                let b = bias[oc];
                for s in 0..spatial {
                    pre[oc * spatial + s] = b;
                }
            }
            self.sparse_conv_accumulate(
                &self.weights[off..off + w_size],
                active_board,
                self.input_channels,
                out_ch,
                &mut pre,
            );
            let mut post = pre.clone();
            for v in post.iter_mut() { *v = v.max(0.0); }
            conv_pre_relu.push(pre);
            conv_post_relu.push(post);
        }

        // ── Conv layers 1+ : dense ──
        for layer_idx in 1..self.conv_channels.len() {
            let in_ch = self.conv_channels[layer_idx - 1];
            let out_ch = self.conv_channels[layer_idx];
            let off = conv_offsets[layer_idx];
            let w_size = out_ch * in_ch * kwpp;
            let bias = &self.weights[off + w_size..off + w_size + out_ch];

            let prev = &conv_post_relu[layer_idx - 1];
            let mut pre = vec![0.0f32; out_ch * spatial];
            for oc in 0..out_ch {
                let b = bias[oc];
                for s in 0..spatial {
                    pre[oc * spatial + s] = b;
                }
            }
            self.dense_conv_accumulate(
                &self.weights[off..off + w_size],
                prev,
                in_ch,
                out_ch,
                &mut pre,
            );
            let mut post = pre.clone();
            for v in post.iter_mut() { *v = v.max(0.0); }
            conv_pre_relu.push(pre);
            conv_post_relu.push(post);
        }

        // ── Flatten + concat bag → FC input ──
        let fc_input_size = self.fc_input_size();
        let last_conv = conv_post_relu.last().unwrap();
        let mut fc_input = Vec::with_capacity(fc_input_size);
        fc_input.extend_from_slice(last_conv);
        fc_input.extend_from_slice(bag);

        // ── FC hidden layers ──
        let fc_offsets = self.fc_layer_offsets();
        let mut fc_pre_relu = Vec::with_capacity(self.fc_sizes.len());
        let mut fc_post_relu = Vec::with_capacity(self.fc_sizes.len());
        let mut prev_act = fc_input;

        for (i, &h) in self.fc_sizes.iter().enumerate() {
            let off = fc_offsets[i];
            let prev_size = prev_act.len();
            let lw = &self.weights[off..off + prev_size * h];
            let lb = &self.weights[off + prev_size * h..off + prev_size * h + h];

            let mut pre = vec![0.0f32; h];
            pre.copy_from_slice(lb);
            for j in 0..prev_size {
                let w_row = &lw[j * h..(j + 1) * h];
                let s = prev_act[j];
                if s != 0.0 {
                    for k in 0..h {
                        pre[k] += w_row[k] * s;
                    }
                }
            }
            let mut post = pre.clone();
            for v in post.iter_mut() { *v = v.max(0.0); }
            fc_pre_relu.push(pre);
            prev_act = post.clone();
            fc_post_relu.push(post);
        }

        // ── Output layer ──
        let out_off = *fc_offsets.last().unwrap();
        let last_h = prev_act.len();
        let out_w = &self.weights[out_off..out_off + last_h];
        let out_b = self.weights[out_off + last_h];

        let mut logit = out_b;
        for j in 0..last_h {
            logit += out_w[j] * prev_act[j];
        }
        let output = sigmoid(logit);

        CnnForwardResult {
            conv_pre_relu,
            conv_post_relu,
            fc_pre_relu,
            fc_post_relu,
            output,
        }
    }

    /// Forward pass with intermediates using pre-allocated scratch buffers.
    /// Returns the output value; intermediate data is stored in `scratch` fields
    /// (conv_pre_relu, conv_post_relu, fc_pre_relu, fc_post_relu).
    pub fn forward_with_intermediates_scratch(
        &self,
        active_board: &[usize],
        bag: &[f32],
        scratch: &mut CnnScratch,
    ) -> f32 {
        let bs = self.board_size;
        let spatial = bs * bs;
        let conv_offsets = &scratch.layout.conv_offsets;
        let kwpp = scratch.layout.kwpp;

        // ── Conv layer 0: sparse input ──
        {
            let out_ch = self.conv_channels[0];
            let off = conv_offsets[0];
            let w_size = out_ch * self.input_channels * kwpp;
            let bias = &self.weights[off + w_size..off + w_size + out_ch];
            let size = out_ch * spatial;

            let pre = &mut scratch.conv_pre_relu[0];
            for v in pre[..size].iter_mut() { *v = 0.0; }
            for oc in 0..out_ch {
                let b = bias[oc];
                for s in 0..spatial {
                    pre[oc * spatial + s] = b;
                }
            }
            self.sparse_conv_accumulate(
                &self.weights[off..off + w_size],
                active_board,
                self.input_channels,
                out_ch,
                &mut pre[..size],
            );
            let post = &mut scratch.conv_post_relu[0];
            post[..size].copy_from_slice(&scratch.conv_pre_relu[0][..size]);
            for v in post[..size].iter_mut() { *v = v.max(0.0); }
        }

        // ── Conv layers 1+ : dense ──
        for layer_idx in 1..self.conv_channels.len() {
            let in_ch = self.conv_channels[layer_idx - 1];
            let out_ch = self.conv_channels[layer_idx];
            let off = conv_offsets[layer_idx];
            let w_size = out_ch * in_ch * kwpp;
            let bias = &self.weights[off + w_size..off + w_size + out_ch];
            let out_size = out_ch * spatial;

            // We need to read conv_post_relu[layer_idx-1] and write conv_pre_relu[layer_idx].
            // Use a pointer to avoid borrow conflicts (both live in scratch).
            let prev_ptr = scratch.conv_post_relu[layer_idx - 1].as_ptr();
            let prev_len = in_ch * spatial;

            let pre = &mut scratch.conv_pre_relu[layer_idx];
            for v in pre[..out_size].iter_mut() { *v = 0.0; }
            for oc in 0..out_ch {
                let b = bias[oc];
                for s in 0..spatial {
                    pre[oc * spatial + s] = b;
                }
            }
            // SAFETY: prev_ptr points to conv_post_relu[layer_idx-1] which is a different
            // Vec from conv_pre_relu[layer_idx]. We only read from prev_ptr.
            let prev_slice = unsafe { std::slice::from_raw_parts(prev_ptr, prev_len) };
            match self.kernel_type {
                KernelType::Box => {
                    Self::dense_conv_im2col_forward(
                        &self.weights[off..off + w_size],
                        prev_slice, in_ch, out_ch, bs,
                        3, 3, 1, 1,
                        &mut scratch.im2col_buf, &mut pre[..out_size],
                    );
                }
                KernelType::Diamond => {
                    Self::dense_conv_im2col_forward(
                        &self.weights[off..off + w_size],
                        prev_slice, in_ch, out_ch, bs,
                        5, 5, 2, 2,
                        &mut scratch.im2col_buf, &mut pre[..out_size],
                    );
                }
                KernelType::Cross => {
                    Self::dense_conv_im2col_forward_cross(
                        &self.weights[off..off + w_size],
                        prev_slice, in_ch, out_ch, bs,
                        &mut scratch.im2col_buf, &mut pre[..out_size],
                    );
                }
            }

            let post = &mut scratch.conv_post_relu[layer_idx];
            post[..out_size].copy_from_slice(&scratch.conv_pre_relu[layer_idx][..out_size]);
            for v in post[..out_size].iter_mut() { *v = v.max(0.0); }
        }

        // ── Flatten + concat bag → fc_input ──
        let fc_input_size = self.fc_input_size();
        let last_ch = *self.conv_channels.last().unwrap();
        let conv_flat_size = last_ch * spatial;
        let last_idx = self.conv_channels.len() - 1;
        scratch.fc_input[..conv_flat_size].copy_from_slice(&scratch.conv_post_relu[last_idx][..conv_flat_size]);
        scratch.fc_input[conv_flat_size..fc_input_size].copy_from_slice(bag);

        // ── FC hidden layers ──
        let fc_offsets = &scratch.layout.fc_offsets;
        // Copy fc_input into fc_prev_act for the first iteration
        scratch.fc_prev_act[..fc_input_size].copy_from_slice(&scratch.fc_input[..fc_input_size]);
        let mut prev_size = fc_input_size;

        for (i, &h) in self.fc_sizes.iter().enumerate() {
            let off = fc_offsets[i];
            let lw = &self.weights[off..off + prev_size * h];
            let lb = &self.weights[off + prev_size * h..off + prev_size * h + h];

            let pre = &mut scratch.fc_pre_relu[i];
            pre[..h].copy_from_slice(lb);
            for j in 0..prev_size {
                let w_row = &lw[j * h..(j + 1) * h];
                let s = scratch.fc_prev_act[j];
                if s != 0.0 {
                    for k in 0..h {
                        pre[k] += w_row[k] * s;
                    }
                }
            }

            let post = &mut scratch.fc_post_relu[i];
            post[..h].copy_from_slice(&scratch.fc_pre_relu[i][..h]);
            for v in post[..h].iter_mut() { *v = v.max(0.0); }

            // Copy post into fc_prev_act for next iteration
            scratch.fc_prev_act[..h].copy_from_slice(&post[..h]);
            prev_size = h;
        }

        // ── Output layer ──
        let out_off = *fc_offsets.last().unwrap();
        let last_h = prev_size;
        let out_w = &self.weights[out_off..out_off + last_h];
        let out_b = self.weights[out_off + last_h];

        let mut logit = out_b;
        for j in 0..last_h {
            logit += out_w[j] * scratch.fc_prev_act[j];
        }

        sigmoid(logit)
    }

    // ── Convolution helpers ───────────────────────────────────────────────

    /// Sparse conv accumulation for the first layer. Only processes active input features.
    ///
    /// Weight layout for the layer:
    ///   For Box:     [out_ch * in_ch * 9]
    ///   For Diamond: [out_ch * in_ch * 25]
    ///   For Cross:   [out_ch * in_ch * 9] (box) + [out_ch * in_ch * 5] (h) + [out_ch * in_ch * 5] (v)
    fn sparse_conv_accumulate(
        &self,
        layer_weights: &[f32],
        active_board: &[usize],
        in_ch: usize,
        out_ch: usize,
        output: &mut [f32],
    ) {
        let bs = self.board_size as i32;
        let spatial = (bs * bs) as usize;

        match self.kernel_type {
            KernelType::Box => {
                // Weight layout: [out_ch][in_ch][3][3]
                // For each active feature at (plane, y, x), it contributes to
                // output positions in a 3x3 neighborhood.
                for &feat_idx in active_board {
                    let plane = feat_idx / spatial;
                    let pos = feat_idx % spatial;
                    let iy = (pos / bs as usize) as i32;
                    let ix = (pos % bs as usize) as i32;

                    for oc in 0..out_ch {
                        // Weight slice for this (oc, ic=plane) pair
                        let w_base = (oc * in_ch + plane) * 9;

                        for ky in 0..3i32 {
                            for kx in 0..3i32 {
                                // Output position that uses input (iy, ix) with kernel offset (ky, kx):
                                // oy = iy - ky + pad, ox = ix - kx + pad, where pad=1
                                let oy = iy - ky + 1;
                                let ox = ix - kx + 1;
                                if oy >= 0 && oy < bs && ox >= 0 && ox < bs {
                                    let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                                    let w_idx = w_base + (ky * 3 + kx) as usize;
                                    output[out_idx] += layer_weights[w_idx];
                                    // input value is 1.0 (binary features)
                                }
                            }
                        }
                    }
                }
            }
            KernelType::Diamond => {
                for &feat_idx in active_board {
                    let plane = feat_idx / spatial;
                    let pos = feat_idx % spatial;
                    let iy = (pos / bs as usize) as i32;
                    let ix = (pos % bs as usize) as i32;

                    for oc in 0..out_ch {
                        let w_base = (oc * in_ch + plane) * 25;
                        for &(ky, kx, k_idx) in &DIAMOND_OFFSETS {
                            let oy = iy - ky + 2;
                            let ox = ix - kx + 2;
                            if oy >= 0 && oy < bs && ox >= 0 && ox < bs {
                                let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                                output[out_idx] += layer_weights[w_base + k_idx];
                            }
                        }
                    }
                }
            }
            KernelType::Cross => {
                // Weight layout: [out_ch * in_ch * 9] (box) then [out_ch * in_ch * 5] (h) then [out_ch * in_ch * 5] (v)
                let box_size = out_ch * in_ch * 9;
                let h_size = out_ch * in_ch * 5;
                let box_w = &layer_weights[..box_size];
                let h_w = &layer_weights[box_size..box_size + h_size];
                let v_w = &layer_weights[box_size + h_size..box_size + h_size + h_size];

                for &feat_idx in active_board {
                    let plane = feat_idx / spatial;
                    let pos = feat_idx % spatial;
                    let iy = (pos / bs as usize) as i32;
                    let ix = (pos % bs as usize) as i32;

                    for oc in 0..out_ch {
                        // Box 3x3, pad=1
                        let bw_base = (oc * in_ch + plane) * 9;
                        for ky in 0..3i32 {
                            for kx in 0..3i32 {
                                let oy = iy - ky + 1;
                                let ox = ix - kx + 1;
                                if oy >= 0 && oy < bs && ox >= 0 && ox < bs {
                                    let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                                    output[out_idx] += box_w[bw_base + (ky * 3 + kx) as usize];
                                }
                            }
                        }

                        // Horizontal 1x5, pad=(0,2)
                        let hw_base = (oc * in_ch + plane) * 5;
                        {
                            let ky = 0i32; // kernel height is 1
                            let oy = iy - ky; // pad_h = 0
                            if oy >= 0 && oy < bs {
                                for kx in 0..5i32 {
                                    let ox = ix - kx + 2; // pad_w = 2
                                    if ox >= 0 && ox < bs {
                                        let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                                        output[out_idx] += h_w[hw_base + kx as usize];
                                    }
                                }
                            }
                        }

                        // Vertical 5x1, pad=(2,0)
                        let vw_base = (oc * in_ch + plane) * 5;
                        for ky in 0..5i32 {
                            let oy = iy - ky + 2; // pad_h = 2
                            let kx = 0i32;
                            let ox = ix - kx; // pad_w = 0
                            if oy >= 0 && oy < bs && ox >= 0 && ox < bs {
                                let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                                output[out_idx] += v_w[vw_base + ky as usize];
                            }
                        }
                    }
                }
            }
        }
    }

    /// Generic dense conv forward via im2col + sgemm.
    ///
    /// Transforms input into im2col matrix, then uses sgemm for the actual convolution.
    /// Supports arbitrary kernel_h x kernel_w with pad_h, pad_w.
    ///
    /// `im2col_buf` must have at least `in_ch * kernel_h * kernel_w * spatial` elements.
    /// `output` is assumed to be pre-initialized with bias values.
    fn dense_conv_im2col_forward(
        layer_weights: &[f32],
        input: &[f32],
        in_ch: usize,
        out_ch: usize,
        bs: usize,
        kernel_h: usize,
        kernel_w: usize,
        pad_h: usize,
        pad_w: usize,
        im2col_buf: &mut [f32],
        output: &mut [f32],
    ) {
        let spatial = bs * bs;
        let kk = kernel_h * kernel_w;
        let k = in_ch * kk; // im2col rows = kernel unrolled size
        let n = spatial; // im2col cols = output spatial positions

        // Build im2col matrix: [in_ch * kernel_h * kernel_w, spatial]
        // Row-major: row r = ic * kk + ky * kernel_w + kx, col c = oy * bs + ox
        // Value = padded_input[ic, oy + ky, ox + kx] (with pad_h, pad_w)
        //
        // The actual input value is input[ic, oy+ky-pad_h, ox+kx-pad_w] when both are in [0, bs)

        // Zero the im2col buffer
        for v in im2col_buf[..k * n].iter_mut() { *v = 0.0; }

        for ic in 0..in_ch {
            let in_base = ic * spatial;
            let col_ic_base = ic * kk;
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let row = col_ic_base + ky * kernel_w + kx;
                    let row_base = row * n;
                    // For output position (oy, ox), we need input at (oy + ky - pad_h, ox + kx - pad_w)
                    // Valid when: oy + ky >= pad_h  and  oy + ky - pad_h < bs
                    let oy_start = if ky >= pad_h { 0 } else { pad_h - ky };
                    let oy_end = bs.min(bs + pad_h - ky);
                    let ox_start = if kx >= pad_w { 0 } else { pad_w - kx };
                    let ox_end = bs.min(bs + pad_w - kx);
                    for oy in oy_start..oy_end {
                        let iy = oy + ky - pad_h;
                        let in_row = in_base + iy * bs;
                        let col_row = row_base + oy * bs;
                        for ox in ox_start..ox_end {
                            let ix = ox + kx - pad_w;
                            im2col_buf[col_row + ox] = input[in_row + ix];
                        }
                    }
                }
            }
        }

        // sgemm: output[out_ch, spatial] += weights[out_ch, k] * im2col[k, spatial]
        // output already contains bias, so beta = 1.0
        unsafe {
            sgemm(
                out_ch,     // m = rows of A (and C)
                k,          // k = cols of A = rows of B
                n,          // n = cols of B (and C)
                1.0,        // alpha
                layer_weights.as_ptr(),  // A: [out_ch x k] row-major
                k as isize, // rsa: row stride of A = k (row-major)
                1,          // csa: col stride of A = 1
                im2col_buf.as_ptr(),     // B: [k x n] row-major
                n as isize, // rsb: row stride of B = n
                1,          // csb: col stride of B = 1
                1.0,        // beta: accumulate into existing output (bias)
                output.as_mut_ptr(),     // C: [out_ch x n] row-major
                n as isize, // rsc: row stride of C = n
                1,          // csc: col stride of C = 1
            );
        }
    }

    /// Dense conv forward via im2col + sgemm for Cross kernel.
    ///
    /// Cross = box(3x3,pad=1) + horizontal(1x5,pad=(0,2)) + vertical(5x1,pad=(2,0)).
    /// Performs 3 im2col+sgemm passes, accumulating into the same output buffer.
    fn dense_conv_im2col_forward_cross(
        layer_weights: &[f32],
        input: &[f32],
        in_ch: usize,
        out_ch: usize,
        bs: usize,
        im2col_buf: &mut [f32],
        output: &mut [f32],
    ) {
        let box_size = out_ch * in_ch * 9;
        let h_size = out_ch * in_ch * 5;

        // Sub-kernel 1: Box 3x3 with pad=1
        Self::dense_conv_im2col_forward(
            &layer_weights[..box_size],
            input, in_ch, out_ch, bs,
            3, 3, 1, 1,
            im2col_buf, output,
        );

        // Sub-kernel 2: Horizontal 1x5 with pad=(0,2)
        Self::dense_conv_im2col_forward(
            &layer_weights[box_size..box_size + h_size],
            input, in_ch, out_ch, bs,
            1, 5, 0, 2,
            im2col_buf, output,
        );

        // Sub-kernel 3: Vertical 5x1 with pad=(2,0)
        Self::dense_conv_im2col_forward(
            &layer_weights[box_size + h_size..box_size + h_size + h_size],
            input, in_ch, out_ch, bs,
            5, 1, 2, 0,
            im2col_buf, output,
        );
    }

    /// Generic dense conv backward via im2col + sgemm.
    ///
    /// Computes weight gradients and input gradients using matrix multiplications.
    /// Supports arbitrary kernel_h x kernel_w with pad_h, pad_w.
    fn dense_conv_im2col_backward(
        layer_weights: &[f32],
        input: &[f32],
        d_pre: &[f32],
        in_ch: usize,
        out_ch: usize,
        bs: usize,
        kernel_h: usize,
        kernel_w: usize,
        pad_h: usize,
        pad_w: usize,
        weight_offset: usize,
        im2col_buf: &mut [f32],
        d_input: &mut [f32],
        grad: &mut [f32],
    ) {
        let spatial = bs * bs;
        let kk = kernel_h * kernel_w;
        let k = in_ch * kk;
        let n = spatial;

        // Step 1: Build im2col from input (same as forward)
        for v in im2col_buf[..k * n].iter_mut() { *v = 0.0; }
        for ic in 0..in_ch {
            let in_base = ic * spatial;
            let col_ic_base = ic * kk;
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let row = col_ic_base + ky * kernel_w + kx;
                    let row_base = row * n;
                    let oy_start = if ky >= pad_h { 0 } else { pad_h - ky };
                    let oy_end = bs.min(bs + pad_h - ky);
                    let ox_start = if kx >= pad_w { 0 } else { pad_w - kx };
                    let ox_end = bs.min(bs + pad_w - kx);
                    for oy in oy_start..oy_end {
                        let iy = oy + ky - pad_h;
                        let in_row = in_base + iy * bs;
                        let col_row = row_base + oy * bs;
                        for ox in ox_start..ox_end {
                            let ix = ox + kx - pad_w;
                            im2col_buf[col_row + ox] = input[in_row + ix];
                        }
                    }
                }
            }
        }

        // Step 2: Weight gradients
        // d_weights[out_ch, k] += d_pre[out_ch, spatial] * im2col^T[spatial, k]
        unsafe {
            sgemm(
                out_ch,     // m
                n,          // k (inner dimension)
                k,          // n (cols of result)
                1.0,        // alpha
                d_pre.as_ptr(),         // A: [out_ch x spatial]
                n as isize, // rsa
                1,          // csa
                im2col_buf.as_ptr(),    // B^T: im2col is [k x n], we want [n x k] = transpose
                1,          // rsb: transposed row stride = original col stride
                n as isize, // csb: transposed col stride = original row stride
                1.0,        // beta: accumulate
                grad[weight_offset..].as_mut_ptr(), // C: weight grads [out_ch x k]
                k as isize, // rsc
                1,          // csc
            );
        }

        // Step 3: Input gradients via col2im
        // d_col[k, spatial] = weights^T[k, out_ch] * d_pre[out_ch, spatial]
        unsafe {
            sgemm(
                k,          // m = rows of result
                out_ch,     // k (inner dimension)
                n,          // n = cols of result
                1.0,        // alpha
                layer_weights.as_ptr(), // A^T: weights is [out_ch x k], we want [k x out_ch]
                1,          // rsa: transposed
                k as isize, // csa: transposed
                d_pre.as_ptr(),         // B: [out_ch x spatial]
                n as isize, // rsb
                1,          // csb
                0.0,        // beta: overwrite
                im2col_buf.as_mut_ptr(), // C: d_col [k x spatial]
                n as isize, // rsc
                1,          // csc
            );
        }

        // col2im: scatter d_col back to d_input
        for ic in 0..in_ch {
            let in_base = ic * spatial;
            let col_ic_base = ic * kk;
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let row = col_ic_base + ky * kernel_w + kx;
                    let row_base = row * n;
                    let oy_start = if ky >= pad_h { 0 } else { pad_h - ky };
                    let oy_end = bs.min(bs + pad_h - ky);
                    let ox_start = if kx >= pad_w { 0 } else { pad_w - kx };
                    let ox_end = bs.min(bs + pad_w - kx);
                    for oy in oy_start..oy_end {
                        let iy = oy + ky - pad_h;
                        let in_row = in_base + iy * bs;
                        let col_row = row_base + oy * bs;
                        for ox in ox_start..ox_end {
                            let ix = ox + kx - pad_w;
                            d_input[in_row + ix] += im2col_buf[col_row + ox];
                        }
                    }
                }
            }
        }
    }

    /// Dense conv backward via im2col + sgemm for Cross kernel.
    ///
    /// Cross = box(3x3,pad=1) + horizontal(1x5,pad=(0,2)) + vertical(5x1,pad=(2,0)).
    /// Performs 3 im2col+sgemm backward passes, accumulating gradients.
    fn dense_conv_im2col_backward_cross(
        layer_weights: &[f32],
        input: &[f32],
        d_pre: &[f32],
        in_ch: usize,
        out_ch: usize,
        bs: usize,
        weight_offset: usize,
        im2col_buf: &mut [f32],
        d_input: &mut [f32],
        grad: &mut [f32],
    ) {
        let box_size = out_ch * in_ch * 9;
        let h_size = out_ch * in_ch * 5;

        // Sub-kernel 1: Box 3x3 with pad=1
        Self::dense_conv_im2col_backward(
            &layer_weights[..box_size],
            input, d_pre, in_ch, out_ch, bs,
            3, 3, 1, 1,
            weight_offset,
            im2col_buf, d_input, grad,
        );

        // Sub-kernel 2: Horizontal 1x5 with pad=(0,2)
        Self::dense_conv_im2col_backward(
            &layer_weights[box_size..box_size + h_size],
            input, d_pre, in_ch, out_ch, bs,
            1, 5, 0, 2,
            weight_offset + box_size,
            im2col_buf, d_input, grad,
        );

        // Sub-kernel 3: Vertical 5x1 with pad=(2,0)
        Self::dense_conv_im2col_backward(
            &layer_weights[box_size + h_size..box_size + h_size + h_size],
            input, d_pre, in_ch, out_ch, bs,
            5, 1, 2, 0,
            weight_offset + box_size + h_size,
            im2col_buf, d_input, grad,
        );
    }

    /// Dense conv accumulation for layers 2+.
    /// Uses pre-padded buffers to eliminate bounds checking from inner loops.
    fn dense_conv_accumulate(
        &self,
        layer_weights: &[f32],
        input: &[f32],
        in_ch: usize,
        out_ch: usize,
        output: &mut [f32],
    ) {
        let bs = self.board_size;
        let spatial = bs * bs;

        match self.kernel_type {
            KernelType::Box => {
                // Pre-pad input: 6x6 -> 8x8 with pad=1
                let pbs = bs + 2; // padded board size = 8
                let pspatial = pbs * pbs; // 64
                // Stack-allocate padded buffer: max 128 channels * 64 = 8192 f32
                let mut padded = [0.0f32; 128 * 64]; // 128 ch * 8*8
                debug_assert!(in_ch <= 128);
                for ic in 0..in_ch {
                    for y in 0..bs {
                        for x in 0..bs {
                            padded[ic * pspatial + (y + 1) * pbs + (x + 1)] =
                                input[ic * spatial + y * bs + x];
                        }
                    }
                }
                for oc in 0..out_ch {
                    for ic in 0..in_ch {
                        let w_base = (oc * in_ch + ic) * 9;
                        let w = &layer_weights[w_base..w_base + 9];
                        let pad_base = ic * pspatial;
                        for oy in 0..bs {
                            let out_row = oc * spatial + oy * bs;
                            let pad_row0 = pad_base + oy * pbs;
                            let pad_row1 = pad_base + (oy + 1) * pbs;
                            let pad_row2 = pad_base + (oy + 2) * pbs;
                            for ox in 0..bs {
                                let sum = padded[pad_row0 + ox] * w[0]
                                    + padded[pad_row0 + ox + 1] * w[1]
                                    + padded[pad_row0 + ox + 2] * w[2]
                                    + padded[pad_row1 + ox] * w[3]
                                    + padded[pad_row1 + ox + 1] * w[4]
                                    + padded[pad_row1 + ox + 2] * w[5]
                                    + padded[pad_row2 + ox] * w[6]
                                    + padded[pad_row2 + ox + 1] * w[7]
                                    + padded[pad_row2 + ox + 2] * w[8];
                                output[out_row + ox] += sum;
                            }
                        }
                    }
                }
            }
            KernelType::Diamond => {
                // Pre-pad input: 6x6 -> 10x10 with pad=2
                let pbs = bs + 4; // 10
                let pspatial = pbs * pbs; // 100
                let mut padded = [0.0f32; 128 * 100]; // 128 ch * 10*10
                debug_assert!(in_ch <= 128);
                for ic in 0..in_ch {
                    for y in 0..bs {
                        for x in 0..bs {
                            padded[ic * pspatial + (y + 2) * pbs + (x + 2)] =
                                input[ic * spatial + y * bs + x];
                        }
                    }
                }
                for oc in 0..out_ch {
                    for ic in 0..in_ch {
                        let w_base = (oc * in_ch + ic) * 25;
                        let pad_base = ic * pspatial;
                        for oy in 0..bs {
                            let out_row = oc * spatial + oy * bs;
                            for ox in 0..bs {
                                let mut sum = 0.0f32;
                                for &(ky, kx, k_idx) in &DIAMOND_OFFSETS {
                                    let py = oy as i32 + ky;
                                    let px = ox as i32 + kx;
                                    sum += padded[pad_base + py as usize * pbs + px as usize]
                                        * layer_weights[w_base + k_idx];
                                }
                                output[out_row + ox] += sum;
                            }
                        }
                    }
                }
            }
            KernelType::Cross => {
                let box_size = out_ch * in_ch * 9;
                let h_size = out_ch * in_ch * 5;
                let box_w = &layer_weights[..box_size];
                let h_w = &layer_weights[box_size..box_size + h_size];
                let v_w = &layer_weights[box_size + h_size..box_size + h_size + h_size];

                // Pre-pad for box (pad=1): 8x8
                let pbs_box = bs + 2;
                let pspatial_box = pbs_box * pbs_box;
                // Pre-pad for horizontal (pad=(0,2)): 6x10
                let ph_w = bs + 4; // width padded to 10
                let ph_spatial = bs * ph_w; // 6*10=60
                // Pre-pad for vertical (pad=(2,0)): 10x6
                let pv_h = bs + 4; // height padded to 10
                let pv_spatial = pv_h * bs; // 10*6=60

                let mut pad_box = [0.0f32; 128 * 64];
                let mut pad_h = [0.0f32; 128 * 60];
                let mut pad_v = [0.0f32; 128 * 60];
                debug_assert!(in_ch <= 128);

                for ic in 0..in_ch {
                    for y in 0..bs {
                        for x in 0..bs {
                            let val = input[ic * spatial + y * bs + x];
                            pad_box[ic * pspatial_box + (y + 1) * pbs_box + (x + 1)] = val;
                            pad_h[ic * ph_spatial + y * ph_w + (x + 2)] = val;
                            pad_v[ic * pv_spatial + (y + 2) * bs + x] = val;
                        }
                    }
                }

                for oc in 0..out_ch {
                    for ic in 0..in_ch {
                        let bw_base = (oc * in_ch + ic) * 9;
                        let hw_base = (oc * in_ch + ic) * 5;
                        let vw_base = (oc * in_ch + ic) * 5;
                        let bw = &box_w[bw_base..bw_base + 9];
                        let hw = &h_w[hw_base..hw_base + 5];
                        let vw = &v_w[vw_base..vw_base + 5];

                        let pb = ic * pspatial_box;
                        let phb = ic * ph_spatial;
                        let pvb = ic * pv_spatial;

                        for oy in 0..bs {
                            let out_row = oc * spatial + oy * bs;
                            // Box padded rows
                            let br0 = pb + oy * pbs_box;
                            let br1 = pb + (oy + 1) * pbs_box;
                            let br2 = pb + (oy + 2) * pbs_box;
                            // Horizontal padded row
                            let hr = phb + oy * ph_w;
                            // Vertical padded rows
                            let vr0 = pvb + oy * bs;
                            let vr1 = pvb + (oy + 1) * bs;
                            let vr2 = pvb + (oy + 2) * bs;
                            let vr3 = pvb + (oy + 3) * bs;
                            let vr4 = pvb + (oy + 4) * bs;

                            for ox in 0..bs {
                                // Box 3x3
                                let sum_box = pad_box[br0 + ox] * bw[0]
                                    + pad_box[br0 + ox + 1] * bw[1]
                                    + pad_box[br0 + ox + 2] * bw[2]
                                    + pad_box[br1 + ox] * bw[3]
                                    + pad_box[br1 + ox + 1] * bw[4]
                                    + pad_box[br1 + ox + 2] * bw[5]
                                    + pad_box[br2 + ox] * bw[6]
                                    + pad_box[br2 + ox + 1] * bw[7]
                                    + pad_box[br2 + ox + 2] * bw[8];

                                // Horizontal 1x5
                                let sum_h = pad_h[hr + ox] * hw[0]
                                    + pad_h[hr + ox + 1] * hw[1]
                                    + pad_h[hr + ox + 2] * hw[2]
                                    + pad_h[hr + ox + 3] * hw[3]
                                    + pad_h[hr + ox + 4] * hw[4];

                                // Vertical 5x1
                                let sum_v = pad_v[vr0 + ox] * vw[0]
                                    + pad_v[vr1 + ox] * vw[1]
                                    + pad_v[vr2 + ox] * vw[2]
                                    + pad_v[vr3 + ox] * vw[3]
                                    + pad_v[vr4 + ox] * vw[4];

                                output[out_row + ox] += sum_box + sum_h + sum_v;
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Save / Load (.gcnn format) ────────────────────────────────────────

    /// Save model to a .gcnn file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"GCNN")?;
        f.write_all(&1u32.to_le_bytes())?; // version
        let kt: u8 = match self.kernel_type {
            KernelType::Box => 0,
            KernelType::Diamond => 1,
            KernelType::Cross => 2,
        };
        f.write_all(&[kt])?;
        f.write_all(&(self.conv_channels.len() as u32).to_le_bytes())?;
        f.write_all(&(self.fc_sizes.len() as u32).to_le_bytes())?;
        f.write_all(&[self.board_size as u8])?;
        f.write_all(&[self.input_channels as u8])?;
        f.write_all(&[self.bag_features as u8])?;
        for &ch in &self.conv_channels {
            f.write_all(&(ch as u32).to_le_bytes())?;
        }
        for &fc in &self.fc_sizes {
            f.write_all(&(fc as u32).to_le_bytes())?;
        }
        for &val in &self.weights {
            f.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load model from a .gcnn file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"GCNN" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Not a GCNN file",
            ));
        }
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported GCNN version {}", version),
            ));
        }
        let mut buf1 = [0u8; 1];
        f.read_exact(&mut buf1)?;
        let kernel_type = match buf1[0] {
            0 => KernelType::Box,
            1 => KernelType::Diamond,
            2 => KernelType::Cross,
            x => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Unknown kernel type {}", x),
                ))
            }
        };
        f.read_exact(&mut buf4)?;
        let num_conv = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let num_fc = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf1)?;
        let board_size = buf1[0] as usize;
        f.read_exact(&mut buf1)?;
        let input_channels = buf1[0] as usize;
        f.read_exact(&mut buf1)?;
        let bag_features = buf1[0] as usize;

        let mut conv_channels = Vec::with_capacity(num_conv);
        for _ in 0..num_conv {
            f.read_exact(&mut buf4)?;
            conv_channels.push(u32::from_le_bytes(buf4) as usize);
        }
        let mut fc_sizes = Vec::with_capacity(num_fc);
        for _ in 0..num_fc {
            f.read_exact(&mut buf4)?;
            fc_sizes.push(u32::from_le_bytes(buf4) as usize);
        }

        let param_count = Self::param_count(
            kernel_type,
            input_channels,
            &conv_channels,
            &fc_sizes,
            board_size,
            bag_features,
        );
        let mut weight_bytes = vec![0u8; param_count * 4];
        f.read_exact(&mut weight_bytes)?;
        let weights: Vec<f32> = weight_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(CnnModel {
            kernel_type,
            conv_channels,
            fc_sizes,
            weights,
            input_channels,
            board_size,
            bag_features,
        })
    }
}

// ── Backward pass ─────────────────────────────────────────────────────────

/// Accumulate gradients for one sample into `grad`.
///
/// MSE loss: L = (output - target)^2.
/// grad layout matches `weights` layout in the model.
pub fn backward(
    model: &CnnModel,
    forward: &CnnForwardResult,
    active_board: &[usize],
    bag: &[f32],
    target: f32,
    sample_weight: f32,
    grad: &mut [f32],
) {
    let conv_offsets = model.conv_layer_offsets();
    let fc_offsets = model.fc_layer_offsets();

    // ── Output gradient: MSE + sigmoid derivative ──
    let o = forward.output;
    let d_logit = 2.0 * (o - target) * o * (1.0 - o) * sample_weight;

    // ── FC backward ──
    let num_fc = model.fc_sizes.len();

    // Output layer gradient
    let out_off = *fc_offsets.last().unwrap();
    let last_h = if num_fc > 0 {
        model.fc_sizes[num_fc - 1]
    } else {
        model.fc_input_size()
    };
    // Build fc_input for gradient computation (needed for first FC layer)
    let fc_input_size = model.fc_input_size();
    let last_conv = &forward.conv_post_relu[forward.conv_post_relu.len() - 1];
    let mut fc_input = Vec::with_capacity(fc_input_size);
    fc_input.extend_from_slice(last_conv);
    fc_input.extend_from_slice(bag);

    if num_fc > 0 {
        // Output layer: weights and bias gradients
        let out_act = &forward.fc_post_relu[num_fc - 1];
        for j in 0..last_h {
            grad[out_off + j] += d_logit * out_act[j];
        }
        grad[out_off + last_h] += d_logit; // bias

        // Backprop through output layer to get d_next for last FC hidden layer
        let out_w = &model.weights[out_off..out_off + last_h];
        let mut d_next: Vec<f32> = (0..last_h).map(|j| d_logit * out_w[j]).collect();

        // FC hidden layers backward
        for layer_idx in (0..num_fc).rev() {
            let cur_size = model.fc_sizes[layer_idx];
            let prev_size = if layer_idx == 0 {
                fc_input_size
            } else {
                model.fc_sizes[layer_idx - 1]
            };
            let off = fc_offsets[layer_idx];
            let bias_off = off + prev_size * cur_size;

            // ReLU derivative
            let mut d_h = vec![0.0f32; cur_size];
            for j in 0..cur_size {
                if forward.fc_pre_relu[layer_idx][j] > 0.0 {
                    d_h[j] = d_next[j];
                }
            }

            // Bias gradient
            for j in 0..cur_size {
                grad[bias_off + j] += d_h[j];
            }

            // Get previous layer activations
            let prev_act = if layer_idx == 0 {
                &fc_input
            } else {
                &forward.fc_post_relu[layer_idx - 1]
            };

            // Weight gradients and propagate
            let lw = &model.weights[off..off + prev_size * cur_size];
            let mut d_prev = vec![0.0f32; prev_size];
            for i in 0..prev_size {
                let w_row = &lw[i * cur_size..(i + 1) * cur_size];
                let mut sum = 0.0f32;
                for j in 0..cur_size {
                    sum += w_row[j] * d_h[j];
                }
                d_prev[i] = sum;
            }
            for i in 0..prev_size {
                let s = prev_act[i];
                if s != 0.0 {
                    for j in 0..cur_size {
                        grad[off + i * cur_size + j] += d_h[j] * s;
                    }
                }
            }

            d_next = d_prev;
        }

        // d_next now has gradient w.r.t. fc_input (size = fc_input_size)
        // Split into conv gradient and bag gradient (bag gradient is discarded)
        let conv_flat_size = fc_input_size - model.bag_features;
        let d_conv_flat = &d_next[..conv_flat_size];
        backward_conv_layers(model, forward, active_board, d_conv_flat, &conv_offsets, grad);
    } else {
        // No FC hidden layers: output layer takes fc_input directly
        for j in 0..fc_input_size {
            grad[out_off + j] += d_logit * fc_input[j];
        }
        grad[out_off + fc_input_size] += d_logit; // bias

        let out_w = &model.weights[out_off..out_off + fc_input_size];
        let d_next: Vec<f32> = (0..fc_input_size).map(|j| d_logit * out_w[j]).collect();

        let conv_flat_size = fc_input_size - model.bag_features;
        let d_conv_flat = &d_next[..conv_flat_size];
        backward_conv_layers(model, forward, active_board, d_conv_flat, &conv_offsets, grad);
    }
}

/// Accumulate gradients for one sample using pre-allocated scratch buffers.
///
/// The scratch must have been filled by a prior call to
/// `forward_with_intermediates_scratch`. `output` is the sigmoid output from that call.
///
/// MSE loss: L = (output - target)^2.
/// grad layout matches `weights` layout in the model.
pub fn backward_scratch(
    model: &CnnModel,
    scratch: &mut CnnScratch,
    output: f32,
    active_board: &[usize],
    bag: &[f32],
    target: f32,
    sample_weight: f32,
    grad: &mut [f32],
) {
    let conv_offsets = model.conv_layer_offsets();
    let fc_offsets = model.fc_layer_offsets();

    // ── Output gradient: MSE + sigmoid derivative ──
    let o = output;
    let d_logit = 2.0 * (o - target) * o * (1.0 - o) * sample_weight;

    // ── FC backward ──
    let num_fc = model.fc_sizes.len();

    let out_off = *fc_offsets.last().unwrap();
    let last_h = if num_fc > 0 {
        model.fc_sizes[num_fc - 1]
    } else {
        model.fc_input_size()
    };

    // Build fc_input in scratch.bk_fc_input
    let fc_input_size = model.fc_input_size();
    let last_conv_idx = scratch.conv_post_relu.len() - 1;
    let conv_flat_size = fc_input_size - model.bag_features;
    scratch.bk_fc_input[..conv_flat_size].copy_from_slice(&scratch.conv_post_relu[last_conv_idx][..conv_flat_size]);
    scratch.bk_fc_input[conv_flat_size..fc_input_size].copy_from_slice(bag);

    if num_fc > 0 {
        // Output layer: weights and bias gradients
        let out_act = &scratch.fc_post_relu[num_fc - 1];
        for j in 0..last_h {
            grad[out_off + j] += d_logit * out_act[j];
        }
        grad[out_off + last_h] += d_logit;

        // d_next = d_logit * out_w  (stored in bk_d_fc_a)
        let out_w = &model.weights[out_off..out_off + last_h];
        for j in 0..last_h {
            scratch.bk_d_fc_a[j] = d_logit * out_w[j];
        }

        // FC hidden layers backward
        for layer_idx in (0..num_fc).rev() {
            let cur_size = model.fc_sizes[layer_idx];
            let prev_size = if layer_idx == 0 {
                fc_input_size
            } else {
                model.fc_sizes[layer_idx - 1]
            };
            let off = fc_offsets[layer_idx];
            let bias_off = off + prev_size * cur_size;

            // ReLU derivative: d_h in bk_d_fc_b
            for j in 0..cur_size {
                scratch.bk_d_fc_b[j] = if scratch.fc_pre_relu[layer_idx][j] > 0.0 {
                    scratch.bk_d_fc_a[j]
                } else {
                    0.0
                };
            }

            // Bias gradient
            for j in 0..cur_size {
                grad[bias_off + j] += scratch.bk_d_fc_b[j];
            }

            // Get previous layer activations
            let prev_act: &[f32] = if layer_idx == 0 {
                &scratch.bk_fc_input[..fc_input_size]
            } else {
                &scratch.fc_post_relu[layer_idx - 1][..prev_size]
            };

            // Weight gradients and propagate: compute d_prev in bk_d_fc_a
            let lw = &model.weights[off..off + prev_size * cur_size];
            // First compute d_prev, writing back into bk_d_fc_a
            // But we're reading bk_d_fc_b (d_h) which is separate, so this is safe
            for i in 0..prev_size {
                let w_row = &lw[i * cur_size..(i + 1) * cur_size];
                let mut sum = 0.0f32;
                for j in 0..cur_size {
                    sum += w_row[j] * scratch.bk_d_fc_b[j];
                }
                scratch.bk_d_fc_a[i] = sum;
            }
            for i in 0..prev_size {
                let s = prev_act[i];
                if s != 0.0 {
                    for j in 0..cur_size {
                        grad[off + i * cur_size + j] += scratch.bk_d_fc_b[j] * s;
                    }
                }
            }
            // bk_d_fc_a[..prev_size] now holds d_next for next iteration
        }

        // bk_d_fc_a[..fc_input_size] has gradient w.r.t. fc_input
        // Copy into bk_d_conv before calling (avoids borrow conflict)
        scratch.bk_d_conv[..conv_flat_size].copy_from_slice(&scratch.bk_d_fc_a[..conv_flat_size]);
        backward_conv_layers_scratch(model, scratch, active_board, conv_flat_size, &conv_offsets, grad);
    } else {
        // No FC hidden layers
        for j in 0..fc_input_size {
            grad[out_off + j] += d_logit * scratch.bk_fc_input[j];
        }
        grad[out_off + fc_input_size] += d_logit;

        let out_w = &model.weights[out_off..out_off + fc_input_size];
        for j in 0..fc_input_size {
            scratch.bk_d_fc_a[j] = d_logit * out_w[j];
        }

        // Copy into bk_d_conv before calling (avoids borrow conflict)
        scratch.bk_d_conv[..conv_flat_size].copy_from_slice(&scratch.bk_d_fc_a[..conv_flat_size]);
        backward_conv_layers_scratch(model, scratch, active_board, conv_flat_size, &conv_offsets, grad);
    }
}

/// Backward pass through conv layers using scratch buffers.
/// Expects scratch.bk_d_conv[..conv_flat_size] to already contain the gradient
/// w.r.t. the flattened last conv layer output.
pub fn backward_conv_layers_scratch(
    model: &CnnModel,
    scratch: &mut CnnScratch,
    active_board: &[usize],
    _conv_flat_size: usize,
    conv_offsets: &[usize],
    grad: &mut [f32],
) {
    let bs = model.board_size;
    let spatial = bs * bs;
    let kwpp = CnnModel::kernel_weights_per_pair(model.kernel_type);
    let num_conv = model.conv_channels.len();

    for layer_idx in (0..num_conv).rev() {
        let out_ch = model.conv_channels[layer_idx];
        let in_ch = if layer_idx == 0 {
            model.input_channels
        } else {
            model.conv_channels[layer_idx - 1]
        };
        let off = conv_offsets[layer_idx];
        let w_size = out_ch * in_ch * kwpp;
        let cur_size = out_ch * spatial;

        // Fuse ReLU derivative with bias gradient
        let pre_relu = &scratch.conv_pre_relu[layer_idx];
        let bias_off = off + w_size;
        let d_post = &mut scratch.bk_d_conv;
        for oc in 0..out_ch {
            let base = oc * spatial;
            let mut sum = 0.0f32;
            for s in 0..spatial {
                let idx = base + s;
                let d = if pre_relu[idx] > 0.0 { d_post[idx] } else { 0.0 };
                d_post[idx] = d;
                sum += d;
            }
            grad[bias_off + oc] += sum;
        }

        if layer_idx == 0 {
            backward_conv_sparse_weights(
                model,
                active_board,
                &scratch.bk_d_conv[..cur_size],
                in_ch,
                out_ch,
                off,
                grad,
            );
        } else {
            let input_data = &scratch.conv_post_relu[layer_idx - 1];
            let in_size = in_ch * spatial;
            // Zero d_input
            for v in scratch.bk_d_input[..in_size].iter_mut() { *v = 0.0; }
            match model.kernel_type {
                KernelType::Box => {
                    CnnModel::dense_conv_im2col_backward(
                        &model.weights[off..off + w_size],
                        input_data,
                        &scratch.bk_d_conv[..cur_size],
                        in_ch, out_ch, bs,
                        3, 3, 1, 1,
                        off,
                        &mut scratch.im2col_buf,
                        &mut scratch.bk_d_input[..in_size],
                        grad,
                    );
                }
                KernelType::Diamond => {
                    CnnModel::dense_conv_im2col_backward(
                        &model.weights[off..off + w_size],
                        input_data,
                        &scratch.bk_d_conv[..cur_size],
                        in_ch, out_ch, bs,
                        5, 5, 2, 2,
                        off,
                        &mut scratch.im2col_buf,
                        &mut scratch.bk_d_input[..in_size],
                        grad,
                    );
                }
                KernelType::Cross => {
                    CnnModel::dense_conv_im2col_backward_cross(
                        &model.weights[off..off + w_size],
                        input_data,
                        &scratch.bk_d_conv[..cur_size],
                        in_ch, out_ch, bs,
                        off,
                        &mut scratch.im2col_buf,
                        &mut scratch.bk_d_input[..in_size],
                        grad,
                    );
                }
            }
            // Copy d_input into d_conv for next iteration
            scratch.bk_d_conv[..in_size].copy_from_slice(&scratch.bk_d_input[..in_size]);
        }
    }
}

/// Backward pass through conv layers. `d_conv_flat` is the gradient w.r.t. the
/// flattened last conv layer output (post-ReLU).
fn backward_conv_layers(
    model: &CnnModel,
    forward: &CnnForwardResult,
    active_board: &[usize],
    d_conv_flat: &[f32],
    conv_offsets: &[usize],
    grad: &mut [f32],
) {
    let bs = model.board_size;
    let spatial = bs * bs;
    let kwpp = CnnModel::kernel_weights_per_pair(model.kernel_type);
    let num_conv = model.conv_channels.len();

    // Start: d_conv_flat is the gradient w.r.t. last conv layer post-ReLU
    let mut d_post = d_conv_flat.to_vec();

    for layer_idx in (0..num_conv).rev() {
        let out_ch = model.conv_channels[layer_idx];
        let in_ch = if layer_idx == 0 {
            model.input_channels
        } else {
            model.conv_channels[layer_idx - 1]
        };
        let off = conv_offsets[layer_idx];
        let w_size = out_ch * in_ch * kwpp;

        // Fuse ReLU derivative with bias gradient computation in a single pass
        // Apply ReLU mask in-place on d_post, then sum for bias
        let pre_relu = &forward.conv_pre_relu[layer_idx];
        let bias_off = off + w_size;
        for oc in 0..out_ch {
            let base = oc * spatial;
            let mut sum = 0.0f32;
            for s in 0..spatial {
                let idx = base + s;
                // Fuse: zero out where pre_relu <= 0 (ReLU derivative), accumulate bias grad
                let d = if pre_relu[idx] > 0.0 { d_post[idx] } else { 0.0 };
                d_post[idx] = d;
                sum += d;
            }
            grad[bias_off + oc] += sum;
        }
        // d_post now contains d_pre (gradient after ReLU derivative applied)

        if layer_idx == 0 {
            // Sparse weight gradient for first layer
            backward_conv_sparse_weights(
                model,
                active_board,
                &d_post,
                in_ch,
                out_ch,
                off,
                grad,
            );
            // No need to propagate gradient to input
        } else {
            // Dense: compute weight gradients and propagate to previous layer
            let input_data = &forward.conv_post_relu[layer_idx - 1];
            let mut d_input = vec![0.0f32; in_ch * spatial];
            backward_conv_dense(
                model,
                &model.weights[off..off + w_size],
                input_data,
                &d_post,
                in_ch,
                out_ch,
                off,
                &mut d_input,
                grad,
            );
            d_post = d_input;
        }
    }
}

/// Sparse weight gradient for conv layer 0.
fn backward_conv_sparse_weights(
    model: &CnnModel,
    active_board: &[usize],
    d_pre: &[f32],
    in_ch: usize,
    out_ch: usize,
    weight_offset: usize,
    grad: &mut [f32],
) {
    let bs = model.board_size as i32;
    let spatial = (bs * bs) as usize;

    match model.kernel_type {
        KernelType::Box => {
            // d_w[oc][ic][ky][kx] += d_out[oc][oy][ox] * input[ic][iy][ix]
            // input is sparse binary: input[plane][iy][ix] = 1.0 only for active features
            for &feat_idx in active_board {
                let plane = feat_idx / spatial;
                let pos = feat_idx % spatial;
                let iy = (pos / bs as usize) as i32;
                let ix = (pos % bs as usize) as i32;

                for oc in 0..out_ch {
                    let w_base = weight_offset + (oc * in_ch + plane) * 9;
                    for ky in 0..3i32 {
                        for kx in 0..3i32 {
                            // oy = iy - ky + pad
                            let oy = iy - ky + 1;
                            let ox = ix - kx + 1;
                            if oy >= 0 && oy < bs && ox >= 0 && ox < bs {
                                let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                                grad[w_base + (ky * 3 + kx) as usize] += d_pre[out_idx];
                                // * 1.0 (input value)
                            }
                        }
                    }
                }
            }
        }
        KernelType::Diamond => {
            for &feat_idx in active_board {
                let plane = feat_idx / spatial;
                let pos = feat_idx % spatial;
                let iy = (pos / bs as usize) as i32;
                let ix = (pos % bs as usize) as i32;

                for oc in 0..out_ch {
                    let w_base = weight_offset + (oc * in_ch + plane) * 25;
                    for &(ky, kx, k_idx) in &DIAMOND_OFFSETS {
                        let oy = iy - ky + 2;
                        let ox = ix - kx + 2;
                        if oy >= 0 && oy < bs && ox >= 0 && ox < bs {
                            let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                            grad[w_base + k_idx] += d_pre[out_idx];
                        }
                    }
                }
            }
        }
        KernelType::Cross => {
            let box_w_off = weight_offset;
            let h_w_off = weight_offset + out_ch * in_ch * 9;
            let v_w_off = h_w_off + out_ch * in_ch * 5;

            for &feat_idx in active_board {
                let plane = feat_idx / spatial;
                let pos = feat_idx % spatial;
                let iy = (pos / bs as usize) as i32;
                let ix = (pos % bs as usize) as i32;

                for oc in 0..out_ch {
                    // Box 3x3
                    let bw_base = box_w_off + (oc * in_ch + plane) * 9;
                    for ky in 0..3i32 {
                        for kx in 0..3i32 {
                            let oy = iy - ky + 1;
                            let ox = ix - kx + 1;
                            if oy >= 0 && oy < bs && ox >= 0 && ox < bs {
                                let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                                grad[bw_base + (ky * 3 + kx) as usize] += d_pre[out_idx];
                            }
                        }
                    }

                    // Horizontal 1x5
                    let hw_base = h_w_off + (oc * in_ch + plane) * 5;
                    {
                        let oy = iy;
                        if oy >= 0 && oy < bs {
                            for kx in 0..5i32 {
                                let ox = ix - kx + 2;
                                if ox >= 0 && ox < bs {
                                    let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                                    grad[hw_base + kx as usize] += d_pre[out_idx];
                                }
                            }
                        }
                    }

                    // Vertical 5x1
                    let vw_base = v_w_off + (oc * in_ch + plane) * 5;
                    for ky in 0..5i32 {
                        let oy = iy - ky + 2;
                        let ox = ix;
                        if oy >= 0 && oy < bs && ox >= 0 && ox < bs {
                            let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                            grad[vw_base + ky as usize] += d_pre[out_idx];
                        }
                    }
                }
            }
        }
    }
}

/// Dense conv backward: compute weight gradients and input gradients.
///
/// Two-pass approach for cache friendliness:
/// 1. Weight gradients: iterate (oy, ox) outer, (oc, ic, kernel) inner — d_pre access is sequential
/// 2. Input gradients: iterate (oc) outer, (oy, ox, kernel) inner — d_input is cache-resident (only 6x6xC)
///
/// Both passes use pre-padded buffers to eliminate bounds checks.
fn backward_conv_dense(
    model: &CnnModel,
    layer_weights: &[f32],
    input: &[f32],
    d_pre: &[f32],
    in_ch: usize,
    out_ch: usize,
    weight_offset: usize,
    d_input: &mut [f32],
    grad: &mut [f32],
) {
    let bs = model.board_size;
    let spatial = bs * bs;

    match model.kernel_type {
        KernelType::Box => {
            // Pre-pad input: 6x6 -> 8x8
            let pbs = bs + 2;
            let pspatial = pbs * pbs;
            let mut padded_in = [0.0f32; 128 * 64];
            debug_assert!(in_ch <= 128);
            for ic in 0..in_ch {
                for y in 0..bs {
                    for x in 0..bs {
                        padded_in[ic * pspatial + (y + 1) * pbs + (x + 1)] =
                            input[ic * spatial + y * bs + x];
                    }
                }
            }

            // Pass 1: weight gradients
            // Iterate (oy, ox) as outer to keep d_pre access sequential
            for oc in 0..out_ch {
                for ic in 0..in_ch {
                    let w_base_grad = weight_offset + (oc * in_ch + ic) * 9;
                    let pad_base = ic * pspatial;
                    let mut wg = [0.0f32; 9];
                    for oy in 0..bs {
                        let out_row = oc * spatial + oy * bs;
                        let pr0 = pad_base + oy * pbs;
                        let pr1 = pad_base + (oy + 1) * pbs;
                        let pr2 = pad_base + (oy + 2) * pbs;
                        for ox in 0..bs {
                            let d = d_pre[out_row + ox];
                            if d == 0.0 { continue; }
                            wg[0] += d * padded_in[pr0 + ox];
                            wg[1] += d * padded_in[pr0 + ox + 1];
                            wg[2] += d * padded_in[pr0 + ox + 2];
                            wg[3] += d * padded_in[pr1 + ox];
                            wg[4] += d * padded_in[pr1 + ox + 1];
                            wg[5] += d * padded_in[pr1 + ox + 2];
                            wg[6] += d * padded_in[pr2 + ox];
                            wg[7] += d * padded_in[pr2 + ox + 1];
                            wg[8] += d * padded_in[pr2 + ox + 2];
                        }
                    }
                    for k in 0..9 {
                        grad[w_base_grad + k] += wg[k];
                    }
                }
            }

            // Pass 2: input gradients
            // Pre-pad d_pre: 6x6 -> 8x8
            let mut padded_d = [0.0f32; 128 * 64];
            debug_assert!(out_ch <= 128);
            for oc_idx in 0..out_ch {
                for y in 0..bs {
                    for x in 0..bs {
                        padded_d[oc_idx * pspatial + (y + 1) * pbs + (x + 1)] =
                            d_pre[oc_idx * spatial + y * bs + x];
                    }
                }
            }
            for oc in 0..out_ch {
                for ic in 0..in_ch {
                    let w_base_local = (oc * in_ch + ic) * 9;
                    let w = &layer_weights[w_base_local..w_base_local + 9];
                    // For input gradient: iy = oy + ky - 1 => oy = iy - ky + 1
                    // Transposed conv: for each input position (iy, ix), sum over kernel
                    // d_input[ic][iy][ix] += sum_ky_kx d_pre[oc][iy-ky+1][ix-kx+1] * w[ky][kx]
                    // With padded d_pre (pad=1), d_pre_pad[iy-ky+1+1][ix-kx+1+1] = d_pre_pad[iy-ky+2][ix-kx+2]
                    // Equivalently: flip kernel and convolve d_pre with flipped kernel
                    let dpad_base = oc * pspatial;
                    for iy in 0..bs {
                        let di_row = ic * spatial + iy * bs;
                        // d_pre_pad rows: iy-0+1=iy+1 down to iy-2+1=iy-1
                        // But we need oy = iy-ky+1, padded_oy = oy+1 = iy-ky+2
                        let dr0 = dpad_base + iy * pbs;       // ky=2: oy=iy-1, pad_oy=iy
                        let dr1 = dpad_base + (iy + 1) * pbs; // ky=1: oy=iy, pad_oy=iy+1
                        let dr2 = dpad_base + (iy + 2) * pbs; // ky=0: oy=iy+1, pad_oy=iy+2
                        for ix in 0..bs {
                            // Transposed convolution = correlation with flipped kernel
                            // d_input[iy][ix] += d_pre[iy+1-ky][ix+1-kx] * w[ky][kx]
                            // With padding shift, padded_d[iy+2-ky][ix+2-kx]
                            let sum = padded_d[dr2 + ix] * w[0]       // ky=0,kx=0
                                + padded_d[dr2 + ix + 1] * w[1]   // ky=0,kx=1
                                + padded_d[dr2 + ix + 2] * w[2]   // ky=0,kx=2
                                + padded_d[dr1 + ix] * w[3]       // ky=1,kx=0
                                + padded_d[dr1 + ix + 1] * w[4]   // ky=1,kx=1
                                + padded_d[dr1 + ix + 2] * w[5]   // ky=1,kx=2
                                + padded_d[dr0 + ix] * w[6]       // ky=2,kx=0
                                + padded_d[dr0 + ix + 1] * w[7]   // ky=2,kx=1
                                + padded_d[dr0 + ix + 2] * w[8];  // ky=2,kx=2
                            d_input[di_row + ix] += sum;
                        }
                    }
                }
            }
        }
        KernelType::Diamond => {
            // Pre-pad input: 6x6 -> 10x10 (pad=2)
            let pbs = bs + 4;
            let pspatial = pbs * pbs;
            let mut padded_in = [0.0f32; 128 * 100];
            debug_assert!(in_ch <= 128);
            for ic in 0..in_ch {
                for y in 0..bs {
                    for x in 0..bs {
                        padded_in[ic * pspatial + (y + 2) * pbs + (x + 2)] =
                            input[ic * spatial + y * bs + x];
                    }
                }
            }

            // Pass 1: weight gradients
            for oc in 0..out_ch {
                for ic in 0..in_ch {
                    let w_base_grad = weight_offset + (oc * in_ch + ic) * 25;
                    let pad_base = ic * pspatial;
                    let mut wg = [0.0f32; 25]; // only 13 will be nonzero
                    for oy in 0..bs {
                        let out_row = oc * spatial + oy * bs;
                        for ox in 0..bs {
                            let d = d_pre[out_row + ox];
                            if d == 0.0 { continue; }
                            for &(ky, kx, k_idx) in &DIAMOND_OFFSETS {
                                let py = oy as i32 + ky;
                                let px = ox as i32 + kx;
                                wg[k_idx] += d * padded_in[pad_base + py as usize * pbs + px as usize];
                            }
                        }
                    }
                    for &(_, _, k_idx) in &DIAMOND_OFFSETS {
                        grad[w_base_grad + k_idx] += wg[k_idx];
                    }
                }
            }

            // Pass 2: input gradients
            // Pre-pad d_pre: 6x6 -> 10x10
            let mut padded_d = [0.0f32; 128 * 100];
            debug_assert!(out_ch <= 128);
            for oc_idx in 0..out_ch {
                for y in 0..bs {
                    for x in 0..bs {
                        padded_d[oc_idx * pspatial + (y + 2) * pbs + (x + 2)] =
                            d_pre[oc_idx * spatial + y * bs + x];
                    }
                }
            }
            // Transposed diamond: for input position (iy,ix), sum over kernel
            // d_input[ic][iy][ix] += sum_{ky,kx in diamond} d_pre[oc][iy-ky+2][ix-kx+2] * w[ky][kx]
            // With padded d_pre (pad=2): d_pre_pad[iy-ky+4][ix-kx+4]
            for oc in 0..out_ch {
                for ic in 0..in_ch {
                    let w_base_local = (oc * in_ch + ic) * 25;
                    let dpad_base = oc * pspatial;
                    for iy in 0..bs {
                        let di_row = ic * spatial + iy * bs;
                        for ix in 0..bs {
                            let mut sum = 0.0f32;
                            for &(ky, kx, k_idx) in &DIAMOND_OFFSETS {
                                // padded position = (iy - ky + 2 + 2, ix - kx + 2 + 2) = (iy - ky + 4, ix - kx + 4)
                                let py = iy as i32 - ky + 4;
                                let px = ix as i32 - kx + 4;
                                sum += padded_d[dpad_base + py as usize * pbs + px as usize]
                                    * layer_weights[w_base_local + k_idx];
                            }
                            d_input[di_row + ix] += sum;
                        }
                    }
                }
            }
        }
        KernelType::Cross => {
            let box_size = out_ch * in_ch * 9;
            let h_size = out_ch * in_ch * 5;
            let box_w = &layer_weights[..box_size];
            let h_w = &layer_weights[box_size..box_size + h_size];
            let v_w = &layer_weights[box_size + h_size..box_size + h_size + h_size];

            let box_grad_off = weight_offset;
            let h_grad_off = weight_offset + box_size;
            let v_grad_off = h_grad_off + h_size;

            // Pre-pad input for box (pad=1): 8x8
            let pbs_box = bs + 2;
            let pspatial_box = pbs_box * pbs_box;
            // Pre-pad input for h (pad=(0,2)): 6x10
            let ph_w = bs + 4;
            let ph_spatial = bs * ph_w;
            // Pre-pad input for v (pad=(2,0)): 10x6
            let pv_h = bs + 4;
            let pv_spatial = pv_h * bs;

            let mut pad_in_box = [0.0f32; 128 * 64];
            let mut pad_in_h = [0.0f32; 128 * 60];
            let mut pad_in_v = [0.0f32; 128 * 60];
            debug_assert!(in_ch <= 128);

            for ic in 0..in_ch {
                for y in 0..bs {
                    for x in 0..bs {
                        let val = input[ic * spatial + y * bs + x];
                        pad_in_box[ic * pspatial_box + (y + 1) * pbs_box + (x + 1)] = val;
                        pad_in_h[ic * ph_spatial + y * ph_w + (x + 2)] = val;
                        pad_in_v[ic * pv_spatial + (y + 2) * bs + x] = val;
                    }
                }
            }

            // Pass 1: weight gradients
            for oc in 0..out_ch {
                for ic in 0..in_ch {
                    let bw_base = (oc * in_ch + ic) * 9;
                    let hw_base = (oc * in_ch + ic) * 5;
                    let vw_base = (oc * in_ch + ic) * 5;

                    let pb = ic * pspatial_box;
                    let phb = ic * ph_spatial;
                    let pvb = ic * pv_spatial;

                    let mut wg_box = [0.0f32; 9];
                    let mut wg_h = [0.0f32; 5];
                    let mut wg_v = [0.0f32; 5];

                    for oy in 0..bs {
                        let out_row = oc * spatial + oy * bs;
                        let br0 = pb + oy * pbs_box;
                        let br1 = pb + (oy + 1) * pbs_box;
                        let br2 = pb + (oy + 2) * pbs_box;
                        let hr = phb + oy * ph_w;
                        let vr0 = pvb + oy * bs;
                        let vr1 = pvb + (oy + 1) * bs;
                        let vr2 = pvb + (oy + 2) * bs;
                        let vr3 = pvb + (oy + 3) * bs;
                        let vr4 = pvb + (oy + 4) * bs;

                        for ox in 0..bs {
                            let d = d_pre[out_row + ox];
                            if d == 0.0 { continue; }

                            // Box 3x3 weight grads
                            wg_box[0] += d * pad_in_box[br0 + ox];
                            wg_box[1] += d * pad_in_box[br0 + ox + 1];
                            wg_box[2] += d * pad_in_box[br0 + ox + 2];
                            wg_box[3] += d * pad_in_box[br1 + ox];
                            wg_box[4] += d * pad_in_box[br1 + ox + 1];
                            wg_box[5] += d * pad_in_box[br1 + ox + 2];
                            wg_box[6] += d * pad_in_box[br2 + ox];
                            wg_box[7] += d * pad_in_box[br2 + ox + 1];
                            wg_box[8] += d * pad_in_box[br2 + ox + 2];

                            // Horizontal 1x5 weight grads
                            wg_h[0] += d * pad_in_h[hr + ox];
                            wg_h[1] += d * pad_in_h[hr + ox + 1];
                            wg_h[2] += d * pad_in_h[hr + ox + 2];
                            wg_h[3] += d * pad_in_h[hr + ox + 3];
                            wg_h[4] += d * pad_in_h[hr + ox + 4];

                            // Vertical 5x1 weight grads
                            wg_v[0] += d * pad_in_v[vr0 + ox];
                            wg_v[1] += d * pad_in_v[vr1 + ox];
                            wg_v[2] += d * pad_in_v[vr2 + ox];
                            wg_v[3] += d * pad_in_v[vr3 + ox];
                            wg_v[4] += d * pad_in_v[vr4 + ox];
                        }
                    }
                    for k in 0..9 { grad[box_grad_off + bw_base + k] += wg_box[k]; }
                    for k in 0..5 { grad[h_grad_off + hw_base + k] += wg_h[k]; }
                    for k in 0..5 { grad[v_grad_off + vw_base + k] += wg_v[k]; }
                }
            }

            // Pass 2: input gradients using pre-padded d_pre
            // Box: pad d_pre with pad=1
            let mut pad_d_box = [0.0f32; 128 * 64];
            // Horizontal: pad d_pre with pad=(0,2)
            let mut pad_d_h = [0.0f32; 128 * 60];
            // Vertical: pad d_pre with pad=(2,0)
            let mut pad_d_v = [0.0f32; 128 * 60];
            debug_assert!(out_ch <= 128);

            for oc_idx in 0..out_ch {
                for y in 0..bs {
                    for x in 0..bs {
                        let val = d_pre[oc_idx * spatial + y * bs + x];
                        pad_d_box[oc_idx * pspatial_box + (y + 1) * pbs_box + (x + 1)] = val;
                        pad_d_h[oc_idx * ph_spatial + y * ph_w + (x + 2)] = val;
                        pad_d_v[oc_idx * pv_spatial + (y + 2) * bs + x] = val;
                    }
                }
            }

            for oc in 0..out_ch {
                for ic in 0..in_ch {
                    let bw_base = (oc * in_ch + ic) * 9;
                    let hw_base = (oc * in_ch + ic) * 5;
                    let vw_base = (oc * in_ch + ic) * 5;
                    let bw = &box_w[bw_base..bw_base + 9];
                    let hw = &h_w[hw_base..hw_base + 5];
                    let vw = &v_w[vw_base..vw_base + 5];

                    let db = oc * pspatial_box;
                    let dhb = oc * ph_spatial;
                    let dvb = oc * pv_spatial;

                    for iy in 0..bs {
                        let di_row = ic * spatial + iy * bs;
                        // Box transposed: padded_d[iy-ky+2][ix-kx+2]
                        let dbr0 = db + iy * pbs_box;
                        let dbr1 = db + (iy + 1) * pbs_box;
                        let dbr2 = db + (iy + 2) * pbs_box;
                        // Horizontal transposed: padded_d_h[iy][ix-kx+2+2]=padded_d_h[iy][ix-kx+4]
                        // d_input[iy][ix] += sum_kx d_pre[iy][ix-kx+2] * hw[kx]
                        // padded: pad_d_h[iy*(bs+4) + ix-kx+2+2] = pad_d_h[iy*(bs+4) + ix-kx+4]
                        let dhr = dhb + iy * ph_w;
                        // Vertical transposed: d_input[iy][ix] += sum_ky d_pre[iy-ky+2][ix] * vw[ky]
                        // padded: pad_d_v[(iy-ky+2+2)*bs + ix] = pad_d_v[(iy-ky+4)*bs + ix]
                        let dvr0 = dvb + iy * bs;
                        let dvr1 = dvb + (iy + 1) * bs;
                        let dvr2 = dvb + (iy + 2) * bs;
                        let dvr3 = dvb + (iy + 3) * bs;
                        let dvr4 = dvb + (iy + 4) * bs;

                        for ix in 0..bs {
                            // Box transposed
                            let sum_box = pad_d_box[dbr2 + ix] * bw[0]
                                + pad_d_box[dbr2 + ix + 1] * bw[1]
                                + pad_d_box[dbr2 + ix + 2] * bw[2]
                                + pad_d_box[dbr1 + ix] * bw[3]
                                + pad_d_box[dbr1 + ix + 1] * bw[4]
                                + pad_d_box[dbr1 + ix + 2] * bw[5]
                                + pad_d_box[dbr0 + ix] * bw[6]
                                + pad_d_box[dbr0 + ix + 1] * bw[7]
                                + pad_d_box[dbr0 + ix + 2] * bw[8];

                            // Horizontal transposed
                            let sum_h = pad_d_h[dhr + ix] * hw[0]
                                + pad_d_h[dhr + ix + 1] * hw[1]
                                + pad_d_h[dhr + ix + 2] * hw[2]
                                + pad_d_h[dhr + ix + 3] * hw[3]
                                + pad_d_h[dhr + ix + 4] * hw[4];

                            // Vertical transposed
                            let sum_v = pad_d_v[dvr4 + ix] * vw[0]
                                + pad_d_v[dvr3 + ix] * vw[1]
                                + pad_d_v[dvr2 + ix] * vw[2]
                                + pad_d_v[dvr1 + ix] * vw[3]
                                + pad_d_v[dvr0 + ix] * vw[4];

                            d_input[di_row + ix] += sum_box + sum_h + sum_v;
                        }
                    }
                }
            }
        }
    }
}

// ── Batched FC forward/backward with sgemm ──────────────────────────────

use matrixmultiply::sgemm;

/// Pre-allocated scratch buffers for batched CNN training.
///
/// Conv layers are per-position (each position has a unique spatial input).
/// FC layers after flatten+concat are batched across positions using sgemm.
///
/// All FC activation/gradient matrices are row-major: [batch_size x neurons].
pub struct CnnBatchScratch {
    /// FC inputs from all positions in the batch: [max_batch x fc_input_size], row-major.
    pub fc_inputs: Vec<f32>,
    /// Per FC hidden layer: pre-ReLU activations [max_batch x layer_size].
    pub fc_pre_act: Vec<Vec<f32>>,
    /// Per FC hidden layer: post-ReLU activations [max_batch x layer_size].
    pub fc_act: Vec<Vec<f32>>,
    /// Per FC hidden layer: gradient [max_batch x layer_size].
    pub fc_d_act: Vec<Vec<f32>>,
    /// Output logits (pre-sigmoid): [max_batch].
    pub logits: Vec<f32>,
    /// Sigmoid outputs: [max_batch].
    pub outputs: Vec<f32>,
    /// d_logit: [max_batch].
    pub d_logits: Vec<f32>,
    /// d_fc_input for backprop into conv: [max_batch x fc_input_size].
    pub d_fc_inputs: Vec<f32>,
    /// Precomputed FC layer weight offsets (within global weight vector).
    pub fc_layer_offsets: Vec<usize>,
    /// FC input size (flattened conv + bag).
    pub fc_input_size: usize,
    /// Per-position conv scratch for forward pass (reused across positions).
    pub conv_scratch: CnnScratch,
    /// Flattened conv intermediates: contiguous buffer for all batch x layer data.
    /// Layout: for each layer L, pre_relu data for all batch positions is at
    ///   offset conv_batch_layer_offsets[L] .. conv_batch_layer_offsets[L] + max_batch * layer_size[L]
    /// Within that: position bi is at conv_batch_layer_offsets[L] + bi * layer_size[L]
    conv_pre_relu_flat: Vec<f32>,
    conv_post_relu_flat: Vec<f32>,
    /// Per-layer offset into conv_pre_relu_flat / conv_post_relu_flat.
    conv_batch_layer_offsets: Vec<usize>,
    /// Backward scratch: d_conv buffer (max conv layer size).
    bk_d_conv: Vec<f32>,
    /// Backward scratch: d_input buffer (max conv layer size, for dense layers).
    bk_d_input: Vec<f32>,
    /// im2col buffer for conv layers: [max_channels * max_kernel_area, spatial].
    im2col_buf: Vec<f32>,
    /// Max batch size (for bounds checking).
    pub max_batch: usize,
    /// Cached layout (offsets + kwpp), computed once to avoid per-call allocations.
    layout: CnnLayout,
}

impl CnnModel {
    /// Create a `CnnBatchScratch` for batched training with the given max batch size.
    pub fn create_batch_scratch(&self, max_batch: usize) -> CnnBatchScratch {
        let fc_input_size = self.fc_input_size();
        let fc_layer_offsets = self.fc_layer_offsets();
        let spatial = self.board_size * self.board_size;

        let fc_pre_act: Vec<Vec<f32>> = self.fc_sizes.iter()
            .map(|&h| vec![0.0f32; max_batch * h])
            .collect();
        let fc_act: Vec<Vec<f32>> = self.fc_sizes.iter()
            .map(|&h| vec![0.0f32; max_batch * h])
            .collect();
        let fc_d_act: Vec<Vec<f32>> = self.fc_sizes.iter()
            .map(|&h| vec![0.0f32; max_batch * h])
            .collect();

        // Flattened conv intermediate storage: contiguous per-layer blocks
        let mut conv_batch_layer_offsets = Vec::with_capacity(self.conv_channels.len());
        let mut total_conv_size = 0usize;
        for &ch in &self.conv_channels {
            conv_batch_layer_offsets.push(total_conv_size);
            let layer_size = ch * spatial;
            total_conv_size += max_batch * layer_size;
        }
        let conv_pre_relu_flat = vec![0.0f32; total_conv_size];
        let conv_post_relu_flat = vec![0.0f32; total_conv_size];

        // Backward scratch buffers
        let max_ch = *self.conv_channels.iter().max().unwrap();
        let max_conv_buf = max_ch * spatial;

        CnnBatchScratch {
            fc_inputs: vec![0.0f32; max_batch * fc_input_size],
            fc_pre_act,
            fc_act,
            fc_d_act,
            logits: vec![0.0f32; max_batch],
            outputs: vec![0.0f32; max_batch],
            d_logits: vec![0.0f32; max_batch],
            d_fc_inputs: vec![0.0f32; max_batch * fc_input_size],
            fc_layer_offsets,
            fc_input_size,
            conv_scratch: self.create_scratch(),
            conv_pre_relu_flat,
            conv_post_relu_flat,
            conv_batch_layer_offsets,
            bk_d_conv: vec![0.0f32; max_conv_buf],
            bk_d_input: vec![0.0f32; max_conv_buf],
            im2col_buf: vec![0.0f32; max_ch * match self.kernel_type {
                KernelType::Box => 9,
                KernelType::Diamond => 25,
                KernelType::Cross => 9,
            } * spatial],
            max_batch,
            layout: CnnLayout::new(self),
        }
    }

    /// Run conv forward for a single position, writing the FC input (flattened conv + bag)
    /// into the appropriate row of `batch_scratch.fc_inputs`.
    ///
    /// Conv intermediates are saved directly into the flat batch buffers for backward use.
    pub fn conv_forward_into_batch(
        &self,
        active_board: &[usize],
        bag: &[f32],
        batch_idx: usize,
        batch_scratch: &mut CnnBatchScratch,
    ) {
        let bs = self.board_size;
        let spatial = bs * bs;
        let conv_offsets = &batch_scratch.layout.conv_offsets;
        let kwpp = batch_scratch.layout.kwpp;

        // ── Conv layer 0: sparse input ──
        {
            let out_ch = self.conv_channels[0];
            let off = conv_offsets[0];
            let w_size = out_ch * self.input_channels * kwpp;
            let bias = &self.weights[off + w_size..off + w_size + out_ch];
            let size = out_ch * spatial;

            // Write pre_relu directly into flat batch buffer
            let layer_off = batch_scratch.conv_batch_layer_offsets[0];
            let pre_start = layer_off + batch_idx * size;
            let pre = &mut batch_scratch.conv_pre_relu_flat[pre_start..pre_start + size];
            for oc in 0..out_ch {
                let b = bias[oc];
                let base = oc * spatial;
                for s in 0..spatial {
                    pre[base + s] = b;
                }
            }
            self.sparse_conv_accumulate(
                &self.weights[off..off + w_size],
                active_board,
                self.input_channels,
                out_ch,
                pre,
            );
            // Fused ReLU: write post_relu into flat batch buffer
            let post_start = layer_off + batch_idx * size;
            let post = &mut batch_scratch.conv_post_relu_flat[post_start..post_start + size];
            for i in 0..size {
                post[i] = batch_scratch.conv_pre_relu_flat[pre_start + i].max(0.0);
            }
        }

        // ── Conv layers 1+ : dense ──
        for layer_idx in 1..self.conv_channels.len() {
            let in_ch = self.conv_channels[layer_idx - 1];
            let out_ch = self.conv_channels[layer_idx];
            let off = conv_offsets[layer_idx];
            let w_size = out_ch * in_ch * kwpp;
            let bias = &self.weights[off + w_size..off + w_size + out_ch];
            let in_size = in_ch * spatial;
            let out_size = out_ch * spatial;

            // Read prev post_relu from flat batch buffer
            let prev_layer_off = batch_scratch.conv_batch_layer_offsets[layer_idx - 1];
            let prev_start = prev_layer_off + batch_idx * in_size;
            let prev_ptr = batch_scratch.conv_post_relu_flat[prev_start..].as_ptr();

            // Write pre_relu directly into flat batch buffer
            let layer_off = batch_scratch.conv_batch_layer_offsets[layer_idx];
            let pre_start = layer_off + batch_idx * out_size;
            let pre = &mut batch_scratch.conv_pre_relu_flat[pre_start..pre_start + out_size];
            for oc in 0..out_ch {
                let b = bias[oc];
                let base = oc * spatial;
                for s in 0..spatial {
                    pre[base + s] = b;
                }
            }
            // SAFETY: prev_ptr points into conv_post_relu_flat, pre points into conv_pre_relu_flat.
            // These are separate Vec allocations so no aliasing.
            let prev_slice = unsafe { std::slice::from_raw_parts(prev_ptr, in_size) };
            match self.kernel_type {
                KernelType::Box => {
                    Self::dense_conv_im2col_forward(
                        &self.weights[off..off + w_size],
                        prev_slice, in_ch, out_ch, bs,
                        3, 3, 1, 1,
                        &mut batch_scratch.im2col_buf, pre,
                    );
                }
                KernelType::Diamond => {
                    Self::dense_conv_im2col_forward(
                        &self.weights[off..off + w_size],
                        prev_slice, in_ch, out_ch, bs,
                        5, 5, 2, 2,
                        &mut batch_scratch.im2col_buf, pre,
                    );
                }
                KernelType::Cross => {
                    Self::dense_conv_im2col_forward_cross(
                        &self.weights[off..off + w_size],
                        prev_slice, in_ch, out_ch, bs,
                        &mut batch_scratch.im2col_buf, pre,
                    );
                }
            }

            // Fused ReLU: write post_relu
            let post_start = layer_off + batch_idx * out_size;
            let post = &mut batch_scratch.conv_post_relu_flat[post_start..post_start + out_size];
            for i in 0..out_size {
                post[i] = batch_scratch.conv_pre_relu_flat[pre_start + i].max(0.0);
            }
        }

        // ── Flatten + concat bag → write into fc_inputs row ���─
        let fc_input_size = batch_scratch.fc_input_size;
        let last_ch = *self.conv_channels.last().unwrap();
        let conv_flat_size = last_ch * spatial;
        let last_idx = self.conv_channels.len() - 1;
        let last_layer_off = batch_scratch.conv_batch_layer_offsets[last_idx];
        let last_start = last_layer_off + batch_idx * conv_flat_size;
        let row_start = batch_idx * fc_input_size;
        batch_scratch.fc_inputs[row_start..row_start + conv_flat_size]
            .copy_from_slice(&batch_scratch.conv_post_relu_flat[last_start..last_start + conv_flat_size]);
        batch_scratch.fc_inputs[row_start + conv_flat_size..row_start + fc_input_size]
            .copy_from_slice(bag);
    }

    /// Batched FC forward pass using sgemm.
    ///
    /// Assumes `batch_scratch.fc_inputs` has been filled by `conv_forward_into_batch`
    /// for all positions in the batch.
    ///
    /// FC layer 0 uses sgemm (input is dense after conv flatten).
    /// Subsequent FC layers also use sgemm.
    /// Output layer: manual dot product (Nx1 output not worth sgemm overhead).
    pub fn batch_fc_forward(
        &self,
        batch_size: usize,
        batch_scratch: &mut CnnBatchScratch,
    ) {
        let num_fc = self.fc_sizes.len();
        let fc_input_size = batch_scratch.fc_input_size;

        if num_fc == 0 {
            // No hidden FC layers: output layer directly from fc_inputs
            let out_off = *batch_scratch.fc_layer_offsets.last().unwrap();
            let out_w = &self.weights[out_off..out_off + fc_input_size];
            let out_b = self.weights[out_off + fc_input_size];

            for bi in 0..batch_size {
                let row = &batch_scratch.fc_inputs[bi * fc_input_size..(bi + 1) * fc_input_size];
                let mut logit = out_b;
                for j in 0..fc_input_size {
                    logit += row[j] * out_w[j];
                }
                batch_scratch.logits[bi] = logit;
                batch_scratch.outputs[bi] = sigmoid(logit);
            }
            return;
        }

        // ── FC layer 0: sgemm (fc_inputs is dense after conv) ���─
        {
            let h = self.fc_sizes[0];
            let off = batch_scratch.fc_layer_offsets[0];
            let lw = &self.weights[off..off + fc_input_size * h];
            let lb = &self.weights[off + fc_input_size * h..off + fc_input_size * h + h];

            // pre_act[0] = fc_inputs x W + bias
            unsafe {
                sgemm(
                    batch_size,                                   // m
                    fc_input_size,                                // k
                    h,                                            // n
                    1.0,                                          // alpha
                    batch_scratch.fc_inputs.as_ptr(),             // A: [batch x fc_input_size]
                    fc_input_size as isize,                       // rsa
                    1,                                            // csa
                    lw.as_ptr(),                                  // B: [fc_input_size x h]
                    h as isize,                                   // rsb
                    1,                                            // csb
                    0.0,                                          // beta
                    batch_scratch.fc_pre_act[0].as_mut_ptr(),     // C: [batch x h]
                    h as isize,                                   // rsc
                    1,                                            // csc
                );
            }

            // Add bias and ReLU
            for bi in 0..batch_size {
                let row_pre = &mut batch_scratch.fc_pre_act[0][bi * h..(bi + 1) * h];
                let row_act = &mut batch_scratch.fc_act[0][bi * h..(bi + 1) * h];
                for j in 0..h {
                    row_pre[j] += lb[j];
                    row_act[j] = row_pre[j].max(0.0);
                }
            }
        }

        // ── Subsequent FC hidden layers: sgemm ──
        for layer_idx in 1..num_fc {
            let prev_size = self.fc_sizes[layer_idx - 1];
            let cur_size = self.fc_sizes[layer_idx];
            let off = batch_scratch.fc_layer_offsets[layer_idx];
            let lw = &self.weights[off..off + prev_size * cur_size];
            let lb = &self.weights[off + prev_size * cur_size..off + prev_size * cur_size + cur_size];

            unsafe {
                sgemm(
                    batch_size,                                        // m
                    prev_size,                                         // k
                    cur_size,                                          // n
                    1.0,                                               // alpha
                    batch_scratch.fc_act[layer_idx - 1].as_ptr(),      // A
                    prev_size as isize,                                // rsa
                    1,                                                 // csa
                    lw.as_ptr(),                                       // B
                    cur_size as isize,                                 // rsb
                    1,                                                 // csb
                    0.0,                                               // beta
                    batch_scratch.fc_pre_act[layer_idx].as_mut_ptr(),  // C
                    cur_size as isize,                                 // rsc
                    1,                                                 // csc
                );
            }

            for bi in 0..batch_size {
                let row_pre = &mut batch_scratch.fc_pre_act[layer_idx][bi * cur_size..(bi + 1) * cur_size];
                let row_act = &mut batch_scratch.fc_act[layer_idx][bi * cur_size..(bi + 1) * cur_size];
                for j in 0..cur_size {
                    row_pre[j] += lb[j];
                    row_act[j] = row_pre[j].max(0.0);
                }
            }
        }

        // ── Output layer: [batch_size x last_hidden] -> [batch_size x 1] ──
        let last_h = self.fc_sizes[num_fc - 1];
        let out_off = *batch_scratch.fc_layer_offsets.last().unwrap();
        let out_w = &self.weights[out_off..out_off + last_h];
        let out_b = self.weights[out_off + last_h];

        let last_act = &batch_scratch.fc_act[num_fc - 1];
        for bi in 0..batch_size {
            let row = &last_act[bi * last_h..(bi + 1) * last_h];
            let mut logit = out_b;
            for j in 0..last_h {
                logit += row[j] * out_w[j];
            }
            batch_scratch.logits[bi] = logit;
            batch_scratch.outputs[bi] = sigmoid(logit);
        }
    }

    /// Batched FC backward pass using sgemm.
    ///
    /// Computes weight gradients for all FC layers and the gradient w.r.t. fc_inputs
    /// (stored in `batch_scratch.d_fc_inputs`) for backprop into conv layers.
    ///
    /// `targets`: [batch_size] target values.
    /// `inv_batch`: 1.0 / batch_size (scaling factor for gradient averaging).
    /// `grad`: weight gradient accumulator (same layout as model.weights).
    pub fn batch_fc_backward(
        &self,
        batch_size: usize,
        targets: &[f32],
        inv_batch: f32,
        batch_scratch: &mut CnnBatchScratch,
        grad: &mut [f32],
    ) {
        let num_fc = self.fc_sizes.len();
        let fc_input_size = batch_scratch.fc_input_size;

        // ── Output gradient: d_logit = 2*(o - t) * o * (1-o) * inv_batch ──
        for bi in 0..batch_size {
            let o = batch_scratch.outputs[bi];
            batch_scratch.d_logits[bi] = 2.0 * (o - targets[bi]) * o * (1.0 - o) * inv_batch;
        }

        if num_fc == 0 {
            // No hidden FC layers: output layer directly from fc_inputs
            let out_off = *batch_scratch.fc_layer_offsets.last().unwrap();

            // Weight gradients
            for bi in 0..batch_size {
                let d = batch_scratch.d_logits[bi];
                let row = &batch_scratch.fc_inputs[bi * fc_input_size..(bi + 1) * fc_input_size];
                for j in 0..fc_input_size {
                    grad[out_off + j] += d * row[j];
                }
                grad[out_off + fc_input_size] += d; // bias
            }

            // d_fc_inputs for conv backward
            let out_w = &self.weights[out_off..out_off + fc_input_size];
            for bi in 0..batch_size {
                let d = batch_scratch.d_logits[bi];
                let row = &mut batch_scratch.d_fc_inputs[bi * fc_input_size..(bi + 1) * fc_input_size];
                for j in 0..fc_input_size {
                    row[j] = d * out_w[j];
                }
            }
            return;
        }

        let last_h = self.fc_sizes[num_fc - 1];
        let out_off = *batch_scratch.fc_layer_offsets.last().unwrap();

        // ── Output layer weight gradients ──
        let last_act = &batch_scratch.fc_act[num_fc - 1];
        let grad_out = &mut grad[out_off..out_off + last_h + 1];
        for bi in 0..batch_size {
            let d = batch_scratch.d_logits[bi];
            let row = &last_act[bi * last_h..(bi + 1) * last_h];
            for j in 0..last_h {
                grad_out[j] += d * row[j];
            }
            grad_out[last_h] += d; // bias
        }

        // ── Backprop d_logit to last hidden layer ──
        let out_w = &self.weights[out_off..out_off + last_h];
        let d_last = &mut batch_scratch.fc_d_act[num_fc - 1];
        for bi in 0..batch_size {
            let d = batch_scratch.d_logits[bi];
            let row = &mut d_last[bi * last_h..(bi + 1) * last_h];
            for j in 0..last_h {
                row[j] = d * out_w[j];
            }
        }

        // Apply ReLU mask for last hidden layer
        let pre_last = &batch_scratch.fc_pre_act[num_fc - 1];
        for i in 0..batch_size * last_h {
            if pre_last[i] <= 0.0 {
                d_last[i] = 0.0;
            }
        }

        // ── FC hidden layers backward (last to second) ──
        for layer_idx in (1..num_fc).rev() {
            let prev_size = self.fc_sizes[layer_idx - 1];
            let cur_size = self.fc_sizes[layer_idx];
            let off = batch_scratch.fc_layer_offsets[layer_idx];
            let bias_offset = off + prev_size * cur_size;
            let lw = &self.weights[off..off + prev_size * cur_size];

            let (d_lower, d_upper) = batch_scratch.fc_d_act.split_at_mut(layer_idx);
            let d_cur = &d_upper[0]; // d_act[layer_idx]
            let d_prev = &mut d_lower[layer_idx - 1]; // d_act[layer_idx - 1]

            // Weight gradient: act[layer-1]^T x d_act[layer]
            let prev_act = &batch_scratch.fc_act[layer_idx - 1];
            unsafe {
                sgemm(
                    prev_size,       // m
                    batch_size,      // k
                    cur_size,        // n
                    1.0,             // alpha
                    prev_act.as_ptr(),   // A (transposed: [batch x prev_size]^T)
                    1,               // rsa
                    prev_size as isize,  // csa
                    d_cur.as_ptr(),      // B
                    cur_size as isize,   // rsb
                    1,               // csb
                    1.0,             // beta: ACCUMULATE into grad
                    grad[off..].as_mut_ptr(), // C
                    cur_size as isize,   // rsc
                    1,               // csc
                );
            }

            // Bias gradient: column sum of d_act[layer]
            for bi in 0..batch_size {
                let row = &d_cur[bi * cur_size..(bi + 1) * cur_size];
                for j in 0..cur_size {
                    grad[bias_offset + j] += row[j];
                }
            }

            // Input gradient: d_act[layer-1] = d_act[layer] x W^T
            unsafe {
                sgemm(
                    batch_size,      // m
                    cur_size,        // k
                    prev_size,       // n
                    1.0,             // alpha
                    d_cur.as_ptr(),      // A
                    cur_size as isize,   // rsa
                    1,               // csa
                    lw.as_ptr(),         // B (transposed)
                    1,               // rsb
                    cur_size as isize,   // csb
                    0.0,             // beta
                    d_prev.as_mut_ptr(), // C
                    prev_size as isize,  // rsc
                    1,               // csc
                );
            }

            // Apply ReLU mask for previous layer
            let pre_prev = &batch_scratch.fc_pre_act[layer_idx - 1];
            for i in 0..batch_size * prev_size {
                if pre_prev[i] <= 0.0 {
                    d_prev[i] = 0.0;
                }
            }
        }

        // ── FC layer 0 backward: gradient w.r.t. fc_inputs ──
        {
            let cur_size = self.fc_sizes[0];
            let off = batch_scratch.fc_layer_offsets[0];
            let bias_offset = off + fc_input_size * cur_size;
            let lw = &self.weights[off..off + fc_input_size * cur_size];
            let d_cur = &batch_scratch.fc_d_act[0];

            // Weight gradient: fc_inputs^T x d_act[0]
            unsafe {
                sgemm(
                    fc_input_size,   // m
                    batch_size,      // k
                    cur_size,        // n
                    1.0,             // alpha
                    batch_scratch.fc_inputs.as_ptr(), // A (transposed)
                    1,               // rsa
                    fc_input_size as isize,           // csa
                    d_cur.as_ptr(),      // B
                    cur_size as isize,   // rsb
                    1,               // csb
                    1.0,             // beta: ACCUMULATE
                    grad[off..].as_mut_ptr(), // C
                    cur_size as isize,   // rsc
                    1,               // csc
                );
            }

            // Bias gradient
            for bi in 0..batch_size {
                let row = &d_cur[bi * cur_size..(bi + 1) * cur_size];
                for j in 0..cur_size {
                    grad[bias_offset + j] += row[j];
                }
            }

            // Input gradient: d_fc_inputs = d_act[0] x W^T
            unsafe {
                sgemm(
                    batch_size,          // m
                    cur_size,            // k
                    fc_input_size,       // n
                    1.0,                 // alpha
                    d_cur.as_ptr(),          // A
                    cur_size as isize,       // rsa
                    1,                   // csa
                    lw.as_ptr(),             // B (transposed)
                    1,                   // rsb
                    cur_size as isize,       // csb
                    0.0,                 // beta
                    batch_scratch.d_fc_inputs.as_mut_ptr(), // C
                    fc_input_size as isize,  // rsc
                    1,                   // csc
                );
            }
        }
    }

    /// Per-position conv backward for one batch element.
    ///
    /// Reads saved conv intermediates directly from the flat batch buffers
    /// (no copying), then runs conv backward in-place.
    pub fn conv_backward_from_batch(
        &self,
        batch_idx: usize,
        active_board: &[usize],
        batch_scratch: &mut CnnBatchScratch,
        grad: &mut [f32],
    ) {
        let bs = self.board_size;
        let spatial = bs * bs;
        let fc_input_size = batch_scratch.fc_input_size;
        let conv_flat_size = fc_input_size - self.bag_features;
        let conv_offsets = &batch_scratch.layout.conv_offsets;
        let kwpp = batch_scratch.layout.kwpp;
        let num_conv = self.conv_channels.len();

        // Copy d_fc_input (conv portion only) into bk_d_conv
        let row_start = batch_idx * fc_input_size;
        batch_scratch.bk_d_conv[..conv_flat_size]
            .copy_from_slice(&batch_scratch.d_fc_inputs[row_start..row_start + conv_flat_size]);

        // Conv backward: iterate layers from last to first
        for layer_idx in (0..num_conv).rev() {
            let out_ch = self.conv_channels[layer_idx];
            let in_ch = if layer_idx == 0 {
                self.input_channels
            } else {
                self.conv_channels[layer_idx - 1]
            };
            let off = conv_offsets[layer_idx];
            let w_size = out_ch * in_ch * kwpp;
            let cur_size = out_ch * spatial;

            // Read pre_relu directly from flat batch buffer for ReLU mask
            let layer_off = batch_scratch.conv_batch_layer_offsets[layer_idx];
            let pre_start = layer_off + batch_idx * cur_size;

            // Fuse ReLU derivative with bias gradient
            let bias_off = off + w_size;
            let d_post = &mut batch_scratch.bk_d_conv;
            for oc in 0..out_ch {
                let base = oc * spatial;
                let mut sum = 0.0f32;
                for s in 0..spatial {
                    let idx = base + s;
                    let d = if batch_scratch.conv_pre_relu_flat[pre_start + idx] > 0.0 { d_post[idx] } else { 0.0 };
                    d_post[idx] = d;
                    sum += d;
                }
                grad[bias_off + oc] += sum;
            }

            if layer_idx == 0 {
                backward_conv_sparse_weights(
                    self,
                    active_board,
                    &batch_scratch.bk_d_conv[..cur_size],
                    in_ch,
                    out_ch,
                    off,
                    grad,
                );
            } else {
                // Read input data (prev layer post_relu) directly from flat batch buffer
                let prev_layer_off = batch_scratch.conv_batch_layer_offsets[layer_idx - 1];
                let prev_size = in_ch * spatial;
                let prev_start = prev_layer_off + batch_idx * prev_size;
                let input_data = &batch_scratch.conv_post_relu_flat[prev_start..prev_start + prev_size];

                // Zero d_input
                for v in batch_scratch.bk_d_input[..prev_size].iter_mut() { *v = 0.0; }
                match self.kernel_type {
                    KernelType::Box => {
                        Self::dense_conv_im2col_backward(
                            &self.weights[off..off + w_size],
                            input_data,
                            &batch_scratch.bk_d_conv[..cur_size],
                            in_ch, out_ch, bs,
                            3, 3, 1, 1,
                            off,
                            &mut batch_scratch.im2col_buf,
                            &mut batch_scratch.bk_d_input[..prev_size],
                            grad,
                        );
                    }
                    KernelType::Diamond => {
                        Self::dense_conv_im2col_backward(
                            &self.weights[off..off + w_size],
                            input_data,
                            &batch_scratch.bk_d_conv[..cur_size],
                            in_ch, out_ch, bs,
                            5, 5, 2, 2,
                            off,
                            &mut batch_scratch.im2col_buf,
                            &mut batch_scratch.bk_d_input[..prev_size],
                            grad,
                        );
                    }
                    KernelType::Cross => {
                        Self::dense_conv_im2col_backward_cross(
                            &self.weights[off..off + w_size],
                            input_data,
                            &batch_scratch.bk_d_conv[..cur_size],
                            in_ch, out_ch, bs,
                            off,
                            &mut batch_scratch.im2col_buf,
                            &mut batch_scratch.bk_d_input[..prev_size],
                            grad,
                        );
                    }
                }
                // Copy d_input into d_conv for next iteration
                batch_scratch.bk_d_conv[..prev_size].copy_from_slice(&batch_scratch.bk_d_input[..prev_size]);
            }
        }
    }
}

// ── CnnEvaluator (for game playing) ─────────────────────────────────────

/// Evaluator wrapper for CnnModel, implementing GameEvaluator.
pub struct CnnEvaluator {
    pub model: CnnModel,
}

impl GameEvaluator for CnnEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        let board = active_board_features(gs);
        let bag = bag_features(gs);
        self.model.forward_sparse(board.as_slice(), &bag)
    }
}

// ── Utility ───────────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Precomputed list of inactive (non-diamond) positions in a 5x5 kernel.
/// These are the 12 corner positions that should always be zero.
const NON_DIAMOND_POSITIONS: [usize; 12] = [0, 1, 3, 4, 5, 9, 15, 19, 20, 21, 23, 24];

/// Apply diamond mask: zero out corner weights in all diamond conv layers.
/// Call after each optimizer step to enforce the mask.
pub fn apply_diamond_mask(model: &mut CnnModel) {
    if model.kernel_type != KernelType::Diamond {
        return;
    }
    let mut offset = 0;
    let mut in_ch = model.input_channels;
    for &out_ch in &model.conv_channels {
        let n_pairs = out_ch * in_ch;
        for pair in 0..n_pairs {
            let base = offset + pair * 25;
            for &kpos in &NON_DIAMOND_POSITIONS {
                model.weights[base + kpos] = 0.0;
            }
        }
        offset += n_pairs * 25 + out_ch; // weights + bias
        in_ch = out_ch;
    }
}

// ── Dense forward for first layer (used in tests) ─────────────────────────

/// Dense forward pass for conv layer 0 (non-sparse). Used in tests to verify
/// that sparse and dense produce identical results.
pub fn forward_dense_conv0(model: &CnnModel, dense_input: &[f32]) -> Vec<f32> {
    let bs = model.board_size as i32;
    let spatial = (bs * bs) as usize;
    let out_ch = model.conv_channels[0];
    let in_ch = model.input_channels;
    let conv_offsets = model.conv_layer_offsets();
    let off = conv_offsets[0];
    let kwpp = CnnModel::kernel_weights_per_pair(model.kernel_type);
    let w_size = out_ch * in_ch * kwpp;
    let bias = &model.weights[off + w_size..off + w_size + out_ch];

    let mut output = vec![0.0f32; out_ch * spatial];
    for oc in 0..out_ch {
        let b = bias[oc];
        for s in 0..spatial {
            output[oc * spatial + s] = b;
        }
    }
    model.dense_conv_accumulate(
        &model.weights[off..off + w_size],
        dense_input,
        in_ch,
        out_ch,
        &mut output,
    );
    output
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{BAG_FEATURES, BOARD_SIZE, NUM_BOARD_PLANES};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn default_board_size() -> usize {
        BOARD_SIZE
    }
    fn default_input_channels() -> usize {
        NUM_BOARD_PLANES
    }
    fn default_bag_features() -> usize {
        BAG_FEATURES
    }

    /// Make a small model for testing.
    fn make_test_model(kernel: KernelType, conv_ch: &[usize], fc: &[usize]) -> CnnModel {
        let mut rng = StdRng::seed_from_u64(42);
        CnnModel::random(
            kernel,
            default_input_channels(),
            conv_ch.to_vec(),
            fc.to_vec(),
            default_board_size(),
            default_bag_features(),
            &mut rng,
        )
    }

    /// Make sample active board features (sparse).
    fn sample_active_board() -> Vec<usize> {
        // A few active features spread across planes
        vec![0, 37, 72, 145, 216, 300, 400, 500, 600, 700, 800, 900, 1000, 1050]
    }

    /// Make sample bag features.
    fn sample_bag() -> Vec<f32> {
        let mut bag = vec![0.0f32; BAG_FEATURES];
        bag[0] = 2.0;
        bag[5] = 1.0;
        bag[13] = 3.0;
        bag[20] = 1.0;
        bag
    }

    // ── Test 1: param_count ──

    #[test]
    fn test_param_count_box() {
        // Conv: 30->8 (3x3): 8*30*9 + 8 = 2168
        // FC input: 8*36 + 26 = 314
        // FC: 314->16: 314*16 + 16 = 5040
        // Output: 16*1 + 1 = 17
        // Total: 2168 + 5040 + 17 = 7225
        let count = CnnModel::param_count(KernelType::Box, 30, &[8], &[16], 6, 26);
        assert_eq!(count, 2168 + 5040 + 17);
    }

    #[test]
    fn test_param_count_diamond() {
        // Conv: 30->8 (5x5): 8*30*25 + 8 = 6008
        // FC input: 8*36 + 26 = 314
        // FC: 314->16: 5040 + 17 = 5057
        // Total: 6008 + 5057 = 11065
        let count = CnnModel::param_count(KernelType::Diamond, 30, &[8], &[16], 6, 26);
        assert_eq!(count, 6008 + 5040 + 17);
    }

    #[test]
    fn test_param_count_cross() {
        // Conv: 30->8 cross: 8*30*(9+5+5) + 8 = 8*30*19 + 8 = 4568
        // FC input: 8*36 + 26 = 314
        // FC: 314->16: 5040 + 17 = 5057
        // Total: 4568 + 5057 = 9625
        let count = CnnModel::param_count(KernelType::Cross, 30, &[8], &[16], 6, 26);
        assert_eq!(count, 4568 + 5040 + 17);
    }

    #[test]
    fn test_param_count_multi_layer() {
        // Conv: 30->64 (box): 64*30*9 + 64 = 17344
        // Conv: 64->32 (box): 32*64*9 + 32 = 18464
        // FC input: 32*36 + 26 = 1178
        // FC: 1178->128: 1178*128 + 128 = 150912
        // Output: 128 + 1 = 129
        // Total: 17344 + 18464 + 150912 + 129 = 186849
        let count = CnnModel::param_count(KernelType::Box, 30, &[64, 32], &[128], 6, 26);
        assert_eq!(count, 17344 + 18464 + 150912 + 129);
    }

    // ── Test 2: forward produces valid output ──

    #[test]
    fn test_forward_valid_output() {
        let model = make_test_model(KernelType::Box, &[8], &[16]);
        let board = sample_active_board();
        let bag = sample_bag();
        let out = model.forward_sparse(&board, &bag);
        assert!(out > 0.0 && out < 1.0, "Output {} not in (0, 1)", out);
    }

    #[test]
    fn test_forward_valid_output_diamond() {
        let model = make_test_model(KernelType::Diamond, &[8], &[16]);
        let board = sample_active_board();
        let bag = sample_bag();
        let out = model.forward_sparse(&board, &bag);
        assert!(out > 0.0 && out < 1.0, "Output {} not in (0, 1)", out);
    }

    #[test]
    fn test_forward_valid_output_cross() {
        let model = make_test_model(KernelType::Cross, &[8], &[16]);
        let board = sample_active_board();
        let bag = sample_bag();
        let out = model.forward_sparse(&board, &bag);
        assert!(out > 0.0 && out < 1.0, "Output {} not in (0, 1)", out);
    }

    // ── Test 3: forward is deterministic ──

    #[test]
    fn test_forward_deterministic() {
        let model = make_test_model(KernelType::Box, &[8, 4], &[16]);
        let board = sample_active_board();
        let bag = sample_bag();
        let out1 = model.forward_sparse(&board, &bag);
        let out2 = model.forward_sparse(&board, &bag);
        assert_eq!(out1, out2, "Forward pass not deterministic");
    }

    // ── Test 4: conv2d_forward correctness with known weights ──

    #[test]
    fn test_conv2d_box_known_weights() {
        // Create a tiny model with all-ones weights, 1 input channel, 1 output channel
        let in_ch = 1;
        let out_ch = 1;
        let bs = 6;
        let spatial = bs * bs;
        let kernel = KernelType::Box;
        // Conv weights: 1*1*9 = 9, all 1.0. Bias: 0.
        // FC: 1*36 + 26 = 62 inputs, 1 hidden neuron, 1 output.
        let fc_sizes = vec![1];
        let param_count = CnnModel::param_count(kernel, in_ch, &[out_ch], &fc_sizes, bs, 26);
        let mut weights = vec![0.0f32; param_count];
        // Set conv weights to 1.0
        for i in 0..9 {
            weights[i] = 1.0;
        }
        // conv bias = 0 (already)
        // FC weights and biases = 0 (already)

        let model = CnnModel {
            kernel_type: kernel,
            conv_channels: vec![out_ch],
            fc_sizes,
            weights,
            input_channels: in_ch,
            board_size: bs,
            bag_features: 26,
        };

        // Dense input: all zeros except one position. input[0][2][3] = 1.0
        let mut dense_input = vec![0.0f32; in_ch * spatial];
        dense_input[2 * bs + 3] = 1.0;

        let conv_out = forward_dense_conv0(&model, &dense_input);

        // With all-1 3x3 kernel and a single 1.0 at (2,3), the output should be 1.0
        // at all 3x3 positions centered around (2,3) that are within bounds:
        // (1,2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,2), (3,3), (3,4)
        for oy in 0..bs {
            for ox in 0..bs {
                let idx = oy * bs + ox;
                let dy = (oy as i32 - 2).abs();
                let dx = (ox as i32 - 3).abs();
                if dy <= 1 && dx <= 1 {
                    assert_eq!(
                        conv_out[idx], 1.0,
                        "Expected 1.0 at ({}, {}), got {}",
                        oy, ox, conv_out[idx]
                    );
                } else {
                    assert_eq!(
                        conv_out[idx], 0.0,
                        "Expected 0.0 at ({}, {}), got {}",
                        oy, ox, conv_out[idx]
                    );
                }
            }
        }
    }

    // ── Test 5: sparse conv1 matches dense conv1 ──

    #[test]
    fn test_sparse_matches_dense_box() {
        test_sparse_matches_dense_kernel(KernelType::Box);
    }

    #[test]
    fn test_sparse_matches_dense_diamond() {
        test_sparse_matches_dense_kernel(KernelType::Diamond);
    }

    #[test]
    fn test_sparse_matches_dense_cross() {
        test_sparse_matches_dense_kernel(KernelType::Cross);
    }

    fn test_sparse_matches_dense_kernel(kernel: KernelType) {
        let model = make_test_model(kernel, &[8], &[16]);
        let active = sample_active_board();
        let bs = model.board_size;
        let spatial = bs * bs;
        let in_ch = model.input_channels;

        // Build dense input from active indices
        let mut dense_input = vec![0.0f32; in_ch * spatial];
        for &idx in &active {
            dense_input[idx] = 1.0;
        }

        // Dense conv0
        let dense_out = forward_dense_conv0(&model, &dense_input);

        // Sparse conv0
        let out_ch = model.conv_channels[0];
        let conv_offsets = model.conv_layer_offsets();
        let off = conv_offsets[0];
        let kwpp = CnnModel::kernel_weights_per_pair(model.kernel_type);
        let w_size = out_ch * in_ch * kwpp;
        let bias = &model.weights[off + w_size..off + w_size + out_ch];

        let mut sparse_out = vec![0.0f32; out_ch * spatial];
        for oc in 0..out_ch {
            let b = bias[oc];
            for s in 0..spatial {
                sparse_out[oc * spatial + s] = b;
            }
        }
        model.sparse_conv_accumulate(
            &model.weights[off..off + w_size],
            &active,
            in_ch,
            out_ch,
            &mut sparse_out,
        );

        // Compare
        for i in 0..dense_out.len() {
            let diff = (dense_out[i] - sparse_out[i]).abs();
            assert!(
                diff < 1e-5,
                "Sparse/dense mismatch at index {}: dense={}, sparse={}, diff={}",
                i, dense_out[i], sparse_out[i], diff
            );
        }
    }

    // ── Test 6: backward gradient check (numerical vs analytical) ──

    #[test]
    fn test_gradient_check_box() {
        gradient_check_kernel(KernelType::Box);
    }

    #[test]
    fn test_gradient_check_diamond() {
        gradient_check_kernel(KernelType::Diamond);
    }

    #[test]
    fn test_gradient_check_cross() {
        gradient_check_kernel(KernelType::Cross);
    }

    fn gradient_check_kernel(kernel: KernelType) {
        let mut model = make_test_model(kernel, &[4], &[8]);
        let active = sample_active_board();
        let bag = sample_bag();
        let target = 0.7f32;
        let n = model.weights.len();

        // Analytical gradient
        let fwd = model.forward_with_intermediates(&active, &bag);
        let mut analytical_grad = vec![0.0f32; n];
        backward(&model, &fwd, &active, &bag, target, 1.0, &mut analytical_grad);

        // Numerical gradient check for a sample of parameters.
        // Use eps=5e-4 for f32 numerical stability (smaller eps amplifies f32 rounding).
        let eps = 5e-4f32;
        let step = (n / 50).max(1);
        let check_indices: Vec<usize> = (0..n).step_by(step).take(50).collect();

        for &i in &check_indices {
            // Skip diamond-masked positions
            if kernel == KernelType::Diamond {
                let mask = diamond_mask_5x5();
                let mut skip = false;
                let mut off = 0;
                let mut ic = model.input_channels;
                for &oc in &model.conv_channels {
                    let w_size = oc * ic * 25;
                    if i >= off && i < off + w_size {
                        let kpos = (i - off) % 25;
                        if !mask[kpos] {
                            assert!(
                                analytical_grad[i].abs() < 1e-6,
                                "Diamond masked param {} has non-zero gradient {}",
                                i, analytical_grad[i]
                            );
                            skip = true;
                        }
                        break;
                    }
                    off += w_size + oc;
                    ic = oc;
                }
                if skip {
                    continue;
                }
            }

            let orig = model.weights[i];

            model.weights[i] = orig + eps;
            let out_plus = model.forward_sparse(&active, &bag);
            let loss_plus = (out_plus - target).powi(2);

            model.weights[i] = orig - eps;
            let out_minus = model.forward_sparse(&active, &bag);
            let loss_minus = (out_minus - target).powi(2);

            model.weights[i] = orig;

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytical = analytical_grad[i];

            let diff = (numerical - analytical).abs();
            let scale = numerical.abs().max(analytical.abs()).max(1e-7);
            let relative = diff / scale;

            assert!(
                relative < 0.05 || diff < 5e-4,
                "Gradient mismatch at param {}: numerical={:.6}, analytical={:.6}, diff={:.6}, relative={:.4}",
                i, numerical, analytical, diff, relative
            );
        }
    }

    // ── Test 7: diamond mask applied correctly ──

    #[test]
    fn test_diamond_mask_applied() {
        let mut model = make_test_model(KernelType::Diamond, &[4, 4], &[8]);
        apply_diamond_mask(&mut model);

        let mask = diamond_mask_5x5();
        let mut offset = 0;
        let mut in_ch = model.input_channels;
        for &out_ch in &model.conv_channels {
            let n_w = out_ch * in_ch * 25;
            for i in 0..n_w {
                let kpos = i % 25;
                if !mask[kpos] {
                    assert_eq!(
                        model.weights[offset + i], 0.0,
                        "Diamond corner weight at offset {} (kpos {}) is not zero: {}",
                        offset + i, kpos, model.weights[offset + i]
                    );
                }
            }
            offset += n_w + out_ch;
            in_ch = out_ch;
        }
    }

    // ── Test 8: cross kernel output is sum of three convolutions ──

    #[test]
    fn test_cross_is_sum_of_three() {
        let mut rng = StdRng::seed_from_u64(123);
        let in_ch = 2;
        let out_ch = 2;
        let bs = 6;
        let spatial = bs * bs;

        // Create a cross model
        let model = CnnModel::random(
            KernelType::Cross,
            in_ch,
            vec![out_ch],
            vec![4],
            bs,
            26,
            &mut rng,
        );

        // Random dense input
        let mut dense_input = vec![0.0f32; in_ch * spatial];
        for v in dense_input.iter_mut() {
            *v = rng.gen::<f32>() * 2.0 - 1.0;
        }

        // Full cross forward
        let cross_out = forward_dense_conv0(&model, &dense_input);

        // Extract weights for separate box, h, v convolutions
        let conv_offsets = model.conv_layer_offsets();
        let off = conv_offsets[0];
        let box_size = out_ch * in_ch * 9;
        let h_size = out_ch * in_ch * 5;
        let v_size = out_ch * in_ch * 5;
        let box_w = &model.weights[off..off + box_size];
        let h_w = &model.weights[off + box_size..off + box_size + h_size];
        let v_w = &model.weights[off + box_size + h_size..off + box_size + h_size + v_size];
        let bias = &model.weights[off + box_size + h_size + v_size..off + box_size + h_size + v_size + out_ch];

        // Manually compute box 3x3
        let mut box_out = vec![0.0f32; out_ch * spatial];
        for oc in 0..out_ch {
            for ic in 0..in_ch {
                let bw = (oc * in_ch + ic) * 9;
                for oy in 0..bs as i32 {
                    for ox in 0..bs as i32 {
                        let mut sum = 0.0f32;
                        for ky in 0..3i32 {
                            for kx in 0..3i32 {
                                let iy = oy + ky - 1;
                                let ix = ox + kx - 1;
                                if iy >= 0 && iy < bs as i32 && ix >= 0 && ix < bs as i32 {
                                    sum += dense_input[ic * spatial + iy as usize * bs + ix as usize]
                                        * box_w[bw + (ky * 3 + kx) as usize];
                                }
                            }
                        }
                        box_out[oc * spatial + oy as usize * bs + ox as usize] += sum;
                    }
                }
            }
        }

        // Manually compute h 1x5
        let mut h_out = vec![0.0f32; out_ch * spatial];
        for oc in 0..out_ch {
            for ic in 0..in_ch {
                let hw = (oc * in_ch + ic) * 5;
                for oy in 0..bs as i32 {
                    for ox in 0..bs as i32 {
                        let mut sum = 0.0f32;
                        let iy = oy; // pad_h=0, ky=0
                        for kx in 0..5i32 {
                            let ix = ox + kx - 2;
                            if ix >= 0 && ix < bs as i32 {
                                sum += dense_input[ic * spatial + iy as usize * bs + ix as usize]
                                    * h_w[hw + kx as usize];
                            }
                        }
                        h_out[oc * spatial + oy as usize * bs + ox as usize] += sum;
                    }
                }
            }
        }

        // Manually compute v 5x1
        let mut v_out = vec![0.0f32; out_ch * spatial];
        for oc in 0..out_ch {
            for ic in 0..in_ch {
                let vw = (oc * in_ch + ic) * 5;
                for oy in 0..bs as i32 {
                    for ox in 0..bs as i32 {
                        let mut sum = 0.0f32;
                        for ky in 0..5i32 {
                            let iy = oy + ky - 2;
                            let ix = ox; // pad_w=0, kx=0
                            if iy >= 0 && iy < bs as i32 {
                                sum += dense_input[ic * spatial + iy as usize * bs + ix as usize]
                                    * v_w[vw + ky as usize];
                            }
                        }
                        v_out[oc * spatial + oy as usize * bs + ox as usize] += sum;
                    }
                }
            }
        }

        // Sum the three + bias
        for oc in 0..out_ch {
            for s in 0..spatial {
                let idx = oc * spatial + s;
                let expected = box_out[idx] + h_out[idx] + v_out[idx] + bias[oc];
                let diff = (cross_out[idx] - expected).abs();
                assert!(
                    diff < 1e-5,
                    "Cross output mismatch at oc={}, s={}: expected={}, got={}, diff={}",
                    oc, s, expected, cross_out[idx], diff
                );
            }
        }
    }

    // ── Test 9: save/load roundtrip ──

    #[test]
    fn test_save_load_roundtrip() {
        let model = make_test_model(KernelType::Box, &[8, 4], &[16]);
        let path = "test_cnn_roundtrip.gcnn";
        model.save(path).expect("Failed to save");

        let loaded = CnnModel::load(path).expect("Failed to load");
        std::fs::remove_file(path).ok();

        assert_eq!(model.kernel_type, loaded.kernel_type);
        assert_eq!(model.conv_channels, loaded.conv_channels);
        assert_eq!(model.fc_sizes, loaded.fc_sizes);
        assert_eq!(model.input_channels, loaded.input_channels);
        assert_eq!(model.board_size, loaded.board_size);
        assert_eq!(model.bag_features, loaded.bag_features);
        assert_eq!(model.weights.len(), loaded.weights.len());
        for (a, b) in model.weights.iter().zip(loaded.weights.iter()) {
            assert_eq!(a, b, "Weight mismatch in save/load roundtrip");
        }
    }

    #[test]
    fn test_save_load_roundtrip_diamond() {
        let model = make_test_model(KernelType::Diamond, &[4], &[8]);
        let path = "test_cnn_roundtrip_diamond.gcnn";
        model.save(path).expect("Failed to save");
        let loaded = CnnModel::load(path).expect("Failed to load");
        std::fs::remove_file(path).ok();
        assert_eq!(model.kernel_type, loaded.kernel_type);
        assert_eq!(model.weights, loaded.weights);
    }

    #[test]
    fn test_save_load_roundtrip_cross() {
        let model = make_test_model(KernelType::Cross, &[4], &[8]);
        let path = "test_cnn_roundtrip_cross.gcnn";
        model.save(path).expect("Failed to save");
        let loaded = CnnModel::load(path).expect("Failed to load");
        std::fs::remove_file(path).ok();
        assert_eq!(model.kernel_type, loaded.kernel_type);
        assert_eq!(model.weights, loaded.weights);
    }

    // ── Test 10: training reduces loss ──

    #[test]
    fn test_training_reduces_loss() {
        let mut model = make_test_model(KernelType::Box, &[4], &[8]);
        let n = model.weights.len();
        let active = sample_active_board();
        let bag = sample_bag();
        let target = 0.7f32;
        let lr = 0.01f32;

        // Compute initial loss
        let initial_out = model.forward_sparse(&active, &bag);
        let initial_loss = (initial_out - target).powi(2);

        // Train for 100 steps
        for _ in 0..100 {
            let fwd = model.forward_with_intermediates(&active, &bag);
            let mut grad = vec![0.0f32; n];
            backward(&model, &fwd, &active, &bag, target, 1.0, &mut grad);
            // SGD update
            for i in 0..n {
                model.weights[i] -= lr * grad[i];
            }
            if model.kernel_type == KernelType::Diamond {
                apply_diamond_mask(&mut model);
            }
        }

        let final_out = model.forward_sparse(&active, &bag);
        let final_loss = (final_out - target).powi(2);

        assert!(
            final_loss < initial_loss,
            "Training did not reduce loss: initial={:.6}, final={:.6}",
            initial_loss, final_loss
        );
        assert!(
            final_loss < 0.05,
            "Loss not small enough after 100 steps: {:.6}",
            final_loss
        );
    }

    #[test]
    fn test_training_reduces_loss_diamond() {
        let mut model = make_test_model(KernelType::Diamond, &[4], &[8]);
        let n = model.weights.len();
        let active = sample_active_board();
        let bag = sample_bag();
        let target = 0.3f32;
        let lr = 0.01f32;

        let initial_out = model.forward_sparse(&active, &bag);
        let initial_loss = (initial_out - target).powi(2);

        for _ in 0..100 {
            let fwd = model.forward_with_intermediates(&active, &bag);
            let mut grad = vec![0.0f32; n];
            backward(&model, &fwd, &active, &bag, target, 1.0, &mut grad);
            for i in 0..n {
                model.weights[i] -= lr * grad[i];
            }
            apply_diamond_mask(&mut model);
        }

        let final_out = model.forward_sparse(&active, &bag);
        let final_loss = (final_out - target).powi(2);

        assert!(
            final_loss < initial_loss,
            "Diamond training did not reduce loss: initial={:.6}, final={:.6}",
            initial_loss, final_loss
        );
    }

    #[test]
    fn test_training_reduces_loss_cross() {
        let mut model = make_test_model(KernelType::Cross, &[4], &[8]);
        let n = model.weights.len();
        let active = sample_active_board();
        let bag = sample_bag();
        let target = 0.6f32;
        let lr = 0.01f32;

        let initial_out = model.forward_sparse(&active, &bag);
        let initial_loss = (initial_out - target).powi(2);

        for _ in 0..100 {
            let fwd = model.forward_with_intermediates(&active, &bag);
            let mut grad = vec![0.0f32; n];
            backward(&model, &fwd, &active, &bag, target, 1.0, &mut grad);
            for i in 0..n {
                model.weights[i] -= lr * grad[i];
            }
        }

        let final_out = model.forward_sparse(&active, &bag);
        let final_loss = (final_out - target).powi(2);

        assert!(
            final_loss < initial_loss,
            "Cross training did not reduce loss: initial={:.6}, final={:.6}",
            initial_loss, final_loss
        );
    }

    // ── Extra: forward_with_intermediates matches forward_sparse ──

    #[test]
    fn test_forward_with_intermediates_matches_sparse() {
        for kernel in [KernelType::Box, KernelType::Diamond, KernelType::Cross] {
            let model = make_test_model(kernel, &[8, 4], &[16]);
            let active = sample_active_board();
            let bag = sample_bag();
            let out_sparse = model.forward_sparse(&active, &bag);
            let fwd = model.forward_with_intermediates(&active, &bag);
            let diff = (out_sparse - fwd.output).abs();
            assert!(
                diff < 1e-5,
                "{:?}: forward_sparse ({}) != forward_with_intermediates ({}), diff={}",
                kernel, out_sparse, fwd.output, diff
            );
        }
    }

    // ── Scratch buffer tests ──

    #[test]
    fn test_forward_sparse_scratch_matches_original() {
        for kernel in [KernelType::Box, KernelType::Diamond, KernelType::Cross] {
            let model = make_test_model(kernel, &[8, 4], &[16]);
            let mut scratch = model.create_scratch();
            let active = sample_active_board();
            let bag = sample_bag();
            let out_orig = model.forward_sparse(&active, &bag);
            let out_scratch = model.forward_sparse_scratch(&active, &bag, &mut scratch);
            let diff = (out_orig - out_scratch).abs();
            assert!(
                diff < 1e-5,
                "{:?}: forward_sparse ({}) != forward_sparse_scratch ({}), diff={}",
                kernel, out_orig, out_scratch, diff
            );
        }
    }

    #[test]
    fn test_forward_with_intermediates_scratch_matches_original() {
        for kernel in [KernelType::Box, KernelType::Diamond, KernelType::Cross] {
            let model = make_test_model(kernel, &[8, 4], &[16]);
            let mut scratch = model.create_scratch();
            let active = sample_active_board();
            let bag = sample_bag();
            let fwd = model.forward_with_intermediates(&active, &bag);
            let out_scratch = model.forward_with_intermediates_scratch(&active, &bag, &mut scratch);
            let diff = (fwd.output - out_scratch).abs();
            assert!(
                diff < 1e-5,
                "{:?}: forward_with_intermediates ({}) != scratch ({}), diff={}",
                kernel, fwd.output, out_scratch, diff
            );
            // Also verify intermediate values match
            for (layer_idx, (orig_pre, scratch_pre)) in fwd.conv_pre_relu.iter().zip(scratch.conv_pre_relu.iter()).enumerate() {
                for (i, (&a, &b)) in orig_pre.iter().zip(scratch_pre.iter()).enumerate() {
                    let d = (a - b).abs();
                    assert!(d < 1e-5, "{:?} layer {} conv_pre_relu[{}]: {} != {}", kernel, layer_idx, i, a, b);
                }
            }
            for (layer_idx, (orig_post, scratch_post)) in fwd.conv_post_relu.iter().zip(scratch.conv_post_relu.iter()).enumerate() {
                for (i, (&a, &b)) in orig_post.iter().zip(scratch_post.iter()).enumerate() {
                    let d = (a - b).abs();
                    assert!(d < 1e-5, "{:?} layer {} conv_post_relu[{}]: {} != {}", kernel, layer_idx, i, a, b);
                }
            }
        }
    }

    #[test]
    fn test_backward_scratch_matches_original() {
        for kernel in [KernelType::Box, KernelType::Diamond, KernelType::Cross] {
            let model = make_test_model(kernel, &[4], &[8]);
            let mut scratch = model.create_scratch();
            let active = sample_active_board();
            let bag = sample_bag();
            let target = 0.7f32;
            let n = model.weights.len();

            // Original backward
            let fwd = model.forward_with_intermediates(&active, &bag);
            let mut grad_orig = vec![0.0f32; n];
            backward(&model, &fwd, &active, &bag, target, 1.0, &mut grad_orig);

            // Scratch backward
            let output = model.forward_with_intermediates_scratch(&active, &bag, &mut scratch);
            let mut grad_scratch = vec![0.0f32; n];
            backward_scratch(&model, &mut scratch, output, &active, &bag, target, 1.0, &mut grad_scratch);

            for i in 0..n {
                let diff = (grad_orig[i] - grad_scratch[i]).abs();
                let scale = grad_orig[i].abs().max(grad_scratch[i].abs()).max(1e-7);
                assert!(
                    diff / scale < 1e-4 || diff < 1e-6,
                    "{:?}: grad mismatch at param {}: orig={:.8}, scratch={:.8}, diff={:.8}",
                    kernel, i, grad_orig[i], grad_scratch[i], diff
                );
            }
        }
    }

    #[test]
    fn test_gradient_check_with_scratch_box() {
        gradient_check_kernel_scratch(KernelType::Box);
    }

    #[test]
    fn test_gradient_check_with_scratch_diamond() {
        gradient_check_kernel_scratch(KernelType::Diamond);
    }

    #[test]
    fn test_gradient_check_with_scratch_cross() {
        gradient_check_kernel_scratch(KernelType::Cross);
    }

    fn gradient_check_kernel_scratch(kernel: KernelType) {
        let mut model = make_test_model(kernel, &[4], &[8]);
        let mut scratch = model.create_scratch();
        let active = sample_active_board();
        let bag = sample_bag();
        let target = 0.7f32;
        let n = model.weights.len();

        // Analytical gradient via scratch
        let output = model.forward_with_intermediates_scratch(&active, &bag, &mut scratch);
        let mut analytical_grad = vec![0.0f32; n];
        backward_scratch(&model, &mut scratch, output, &active, &bag, target, 1.0, &mut analytical_grad);

        let eps = 5e-4f32;
        let step = (n / 50).max(1);
        let check_indices: Vec<usize> = (0..n).step_by(step).take(50).collect();

        for &i in &check_indices {
            if kernel == KernelType::Diamond {
                let mask = diamond_mask_5x5();
                let mut skip = false;
                let mut off = 0;
                let mut ic = model.input_channels;
                for &oc in &model.conv_channels {
                    let w_size = oc * ic * 25;
                    if i >= off && i < off + w_size {
                        let kpos = (i - off) % 25;
                        if !mask[kpos] {
                            skip = true;
                        }
                        break;
                    }
                    off += w_size + oc;
                    ic = oc;
                }
                if skip { continue; }
            }

            let orig = model.weights[i];
            model.weights[i] = orig + eps;
            let out_plus = model.forward_sparse(&active, &bag);
            let loss_plus = (out_plus - target).powi(2);
            model.weights[i] = orig - eps;
            let out_minus = model.forward_sparse(&active, &bag);
            let loss_minus = (out_minus - target).powi(2);
            model.weights[i] = orig;

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytical = analytical_grad[i];
            let diff = (numerical - analytical).abs();
            let scale = numerical.abs().max(analytical.abs()).max(1e-7);
            let relative = diff / scale;

            assert!(
                relative < 0.05 || diff < 5e-4,
                "Scratch gradient mismatch at param {}: numerical={:.6}, analytical={:.6}, diff={:.6}, relative={:.4}",
                i, numerical, analytical, diff, relative
            );
        }
    }

    #[test]
    fn test_scratch_reuse_across_positions() {
        // Verify that reusing scratch produces correct results for different positions
        let model = make_test_model(KernelType::Box, &[8, 4], &[16]);
        let mut scratch = model.create_scratch();

        let boards = vec![
            vec![0, 37, 72],
            vec![145, 216, 300, 400, 500],
            vec![600, 700, 800, 900, 1000, 1050],
        ];
        let bags = vec![
            sample_bag(),
            {
                let mut b = vec![0.0f32; BAG_FEATURES];
                b[3] = 5.0;
                b[10] = 2.0;
                b
            },
            vec![1.0f32; BAG_FEATURES],
        ];

        for (board, bag) in boards.iter().zip(bags.iter()) {
            let out_orig = model.forward_sparse(board, bag);
            let out_scratch = model.forward_sparse_scratch(board, bag, &mut scratch);
            let diff = (out_orig - out_scratch).abs();
            assert!(
                diff < 1e-5,
                "Scratch reuse mismatch: orig={}, scratch={}, diff={}",
                out_orig, out_scratch, diff
            );
        }
    }

    // ── Test: batched FC forward matches per-position forward ──

    #[test]
    fn test_batch_fc_forward_matches_per_position() {
        test_batch_fc_matches_kernel(KernelType::Box);
    }

    #[test]
    fn test_batch_fc_forward_matches_per_position_diamond() {
        test_batch_fc_matches_kernel(KernelType::Diamond);
    }

    #[test]
    fn test_batch_fc_forward_matches_per_position_cross() {
        test_batch_fc_matches_kernel(KernelType::Cross);
    }

    fn test_batch_fc_matches_kernel(kernel: KernelType) {
        let model = make_test_model(kernel, &[8, 4], &[16, 8]);
        let boards = vec![
            sample_active_board(),
            vec![10, 50, 200, 500, 800, 1050],
            vec![0, 36, 72, 108, 144, 180, 216, 360],
        ];
        let bags = vec![
            sample_bag(),
            {
                let mut b = vec![0.0f32; BAG_FEATURES];
                b[3] = 5.0; b[10] = 2.0;
                b
            },
            vec![1.0f32; BAG_FEATURES],
        ];
        let batch_size = boards.len();

        // Per-position forward using existing scratch-based method
        let mut scratch = model.create_scratch();
        let per_pos_outputs: Vec<f32> = boards.iter().zip(bags.iter())
            .map(|(board, bag)| model.forward_with_intermediates_scratch(board, bag, &mut scratch))
            .collect();

        // Batched forward
        let mut batch_scratch = model.create_batch_scratch(batch_size);
        for (bi, (board, bag)) in boards.iter().zip(bags.iter()).enumerate() {
            model.conv_forward_into_batch(board, bag, bi, &mut batch_scratch);
        }
        model.batch_fc_forward(batch_size, &mut batch_scratch);

        // Compare outputs
        for bi in 0..batch_size {
            let diff = (per_pos_outputs[bi] - batch_scratch.outputs[bi]).abs();
            assert!(
                diff < 1e-5,
                "Batch vs per-position output mismatch at position {}: per_pos={}, batch={}, diff={}",
                bi, per_pos_outputs[bi], batch_scratch.outputs[bi], diff
            );
        }
    }

    // ── Test: batched FC backward matches per-position backward ──

    #[test]
    fn test_batch_fc_backward_matches_per_position() {
        test_batch_backward_matches_kernel(KernelType::Box);
    }

    #[test]
    fn test_batch_fc_backward_matches_per_position_diamond() {
        test_batch_backward_matches_kernel(KernelType::Diamond);
    }

    #[test]
    fn test_batch_fc_backward_matches_per_position_cross() {
        test_batch_backward_matches_kernel(KernelType::Cross);
    }

    fn test_batch_backward_matches_kernel(kernel: KernelType) {
        let model = make_test_model(kernel, &[8, 4], &[16, 8]);
        let num_params = model.weights.len();
        let boards = vec![
            sample_active_board(),
            vec![10, 50, 200, 500, 800, 1050],
            vec![0, 36, 72, 108, 144, 180, 216, 360],
        ];
        let bags = vec![
            sample_bag(),
            {
                let mut b = vec![0.0f32; BAG_FEATURES];
                b[3] = 5.0; b[10] = 2.0;
                b
            },
            vec![1.0f32; BAG_FEATURES],
        ];
        let targets = vec![0.3f32, 0.7, 0.5];
        let batch_size = boards.len();
        let inv_batch = 1.0 / batch_size as f32;

        // Per-position forward+backward
        let mut per_pos_grad = vec![0.0f32; num_params];
        let mut scratch = model.create_scratch();
        for ((board, bag), &target) in boards.iter().zip(bags.iter()).zip(targets.iter()) {
            let output = model.forward_with_intermediates_scratch(board, bag, &mut scratch);
            backward_scratch(
                &model,
                &mut scratch,
                output,
                board,
                bag,
                target,
                inv_batch,
                &mut per_pos_grad,
            );
        }

        // Batched forward+backward
        let mut batch_grad = vec![0.0f32; num_params];
        let mut batch_scratch = model.create_batch_scratch(batch_size);
        for (bi, (board, bag)) in boards.iter().zip(bags.iter()).enumerate() {
            model.conv_forward_into_batch(board, bag, bi, &mut batch_scratch);
        }
        model.batch_fc_forward(batch_size, &mut batch_scratch);
        model.batch_fc_backward(batch_size, &targets, inv_batch, &mut batch_scratch, &mut batch_grad);
        for bi in 0..batch_size {
            model.conv_backward_from_batch(bi, &boards[bi], &mut batch_scratch, &mut batch_grad);
        }

        // Compare gradients
        let mut max_diff = 0.0f32;
        let mut max_diff_idx = 0;
        for i in 0..num_params {
            let diff = (per_pos_grad[i] - batch_grad[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
        }
        assert!(
            max_diff < 1e-3,
            "Gradient mismatch at index {}: per_pos={}, batch={}, diff={}",
            max_diff_idx, per_pos_grad[max_diff_idx], batch_grad[max_diff_idx], max_diff
        );
    }
}
