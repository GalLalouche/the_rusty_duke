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

// ── Weight layout helpers ─────────────────────────────────────────────────

impl CnnModel {
    /// Kernel footprint (number of f32 weights per (out_ch, in_ch) pair) for a single conv.
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
    pub fn fc_input_size(&self) -> usize {
        self.conv_channels.last().unwrap() * self.board_size * self.board_size + self.bag_features
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
            let post: Vec<f32> = pre.iter().map(|&v| v.max(0.0)).collect();
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
            let post: Vec<f32> = pre.iter().map(|&v| v.max(0.0)).collect();
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
            let post: Vec<f32> = pre.iter().map(|&v| v.max(0.0)).collect();
            fc_pre_relu.push(pre);
            fc_post_relu.push(post.clone());
            prev_act = post;
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
                let mask = diamond_mask_5x5();
                for &feat_idx in active_board {
                    let plane = feat_idx / spatial;
                    let pos = feat_idx % spatial;
                    let iy = (pos / bs as usize) as i32;
                    let ix = (pos % bs as usize) as i32;

                    for oc in 0..out_ch {
                        let w_base = (oc * in_ch + plane) * 25;
                        for ky in 0..5i32 {
                            for kx in 0..5i32 {
                                let k_idx = (ky * 5 + kx) as usize;
                                if !mask[k_idx] {
                                    continue;
                                }
                                // pad=2
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

    /// Dense conv accumulation for layers 2+.
    fn dense_conv_accumulate(
        &self,
        layer_weights: &[f32],
        input: &[f32],
        in_ch: usize,
        out_ch: usize,
        output: &mut [f32],
    ) {
        let bs = self.board_size as i32;
        let spatial = (bs * bs) as usize;

        match self.kernel_type {
            KernelType::Box => {
                for oc in 0..out_ch {
                    for ic in 0..in_ch {
                        let w_base = (oc * in_ch + ic) * 9;
                        for oy in 0..bs {
                            for ox in 0..bs {
                                let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                                let mut sum = 0.0f32;
                                for ky in 0..3i32 {
                                    for kx in 0..3i32 {
                                        let iy = oy + ky - 1; // pad=1
                                        let ix = ox + kx - 1;
                                        if iy >= 0 && iy < bs && ix >= 0 && ix < bs {
                                            let in_idx = ic * spatial + (iy as usize) * bs as usize + ix as usize;
                                            sum += input[in_idx] * layer_weights[w_base + (ky * 3 + kx) as usize];
                                        }
                                    }
                                }
                                output[out_idx] += sum;
                            }
                        }
                    }
                }
            }
            KernelType::Diamond => {
                let mask = diamond_mask_5x5();
                for oc in 0..out_ch {
                    for ic in 0..in_ch {
                        let w_base = (oc * in_ch + ic) * 25;
                        for oy in 0..bs {
                            for ox in 0..bs {
                                let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                                let mut sum = 0.0f32;
                                for ky in 0..5i32 {
                                    for kx in 0..5i32 {
                                        let k_idx = (ky * 5 + kx) as usize;
                                        if !mask[k_idx] {
                                            continue;
                                        }
                                        let iy = oy + ky - 2; // pad=2
                                        let ix = ox + kx - 2;
                                        if iy >= 0 && iy < bs && ix >= 0 && ix < bs {
                                            let in_idx = ic * spatial + (iy as usize) * bs as usize + ix as usize;
                                            sum += input[in_idx] * layer_weights[w_base + k_idx];
                                        }
                                    }
                                }
                                output[out_idx] += sum;
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

                for oc in 0..out_ch {
                    for ic in 0..in_ch {
                        let bw_base = (oc * in_ch + ic) * 9;
                        let hw_base = (oc * in_ch + ic) * 5;
                        let vw_base = (oc * in_ch + ic) * 5;

                        for oy in 0..bs {
                            for ox in 0..bs {
                                let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                                let mut sum = 0.0f32;

                                // Box 3x3, pad=1
                                for ky in 0..3i32 {
                                    for kx in 0..3i32 {
                                        let iy = oy + ky - 1;
                                        let ix = ox + kx - 1;
                                        if iy >= 0 && iy < bs && ix >= 0 && ix < bs {
                                            let in_idx = ic * spatial + (iy as usize) * bs as usize + ix as usize;
                                            sum += input[in_idx] * box_w[bw_base + (ky * 3 + kx) as usize];
                                        }
                                    }
                                }

                                // Horizontal 1x5, pad=(0,2)
                                {
                                    let iy = oy; // pad_h = 0, ky=0
                                    if iy >= 0 && iy < bs {
                                        for kx in 0..5i32 {
                                            let ix = ox + kx - 2;
                                            if ix >= 0 && ix < bs {
                                                let in_idx = ic * spatial + (iy as usize) * bs as usize + ix as usize;
                                                sum += input[in_idx] * h_w[hw_base + kx as usize];
                                            }
                                        }
                                    }
                                }

                                // Vertical 5x1, pad=(2,0)
                                for ky in 0..5i32 {
                                    let iy = oy + ky - 2;
                                    let ix = ox; // pad_w = 0, kx=0
                                    if iy >= 0 && iy < bs && ix >= 0 && ix < bs {
                                        let in_idx = ic * spatial + (iy as usize) * bs as usize + ix as usize;
                                        sum += input[in_idx] * v_w[vw_base + ky as usize];
                                    }
                                }

                                output[out_idx] += sum;
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
    let bs = model.board_size as i32;
    let spatial = (bs * bs) as usize;
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

        // Apply ReLU derivative
        let mut d_pre = vec![0.0f32; out_ch * spatial];
        for i in 0..out_ch * spatial {
            if forward.conv_pre_relu[layer_idx][i] > 0.0 {
                d_pre[i] = d_post[i];
            }
        }

        // Bias gradient: sum over spatial dimensions for each output channel
        let bias_off = off + w_size;
        for oc in 0..out_ch {
            let mut sum = 0.0f32;
            for s in 0..spatial {
                sum += d_pre[oc * spatial + s];
            }
            grad[bias_off + oc] += sum;
        }

        // Get input to this layer
        let input_data: &[f32] = if layer_idx > 0 {
            &forward.conv_post_relu[layer_idx - 1]
        } else {
            &[] // sparse — handled differently
        };

        if layer_idx == 0 {
            // Sparse weight gradient for first layer
            backward_conv_sparse_weights(
                model,
                active_board,
                &d_pre,
                in_ch,
                out_ch,
                off,
                grad,
            );
            // No need to propagate gradient to input
        } else {
            // Dense: compute weight gradients and propagate to previous layer
            let mut d_input = vec![0.0f32; in_ch * spatial];
            backward_conv_dense(
                model,
                &model.weights[off..off + w_size],
                input_data,
                &d_pre,
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
            let mask = diamond_mask_5x5();
            for &feat_idx in active_board {
                let plane = feat_idx / spatial;
                let pos = feat_idx % spatial;
                let iy = (pos / bs as usize) as i32;
                let ix = (pos % bs as usize) as i32;

                for oc in 0..out_ch {
                    let w_base = weight_offset + (oc * in_ch + plane) * 25;
                    for ky in 0..5i32 {
                        for kx in 0..5i32 {
                            let k_idx = (ky * 5 + kx) as usize;
                            if !mask[k_idx] {
                                continue;
                            }
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
    let bs = model.board_size as i32;
    let spatial = (bs * bs) as usize;

    match model.kernel_type {
        KernelType::Box => {
            for oc in 0..out_ch {
                for ic in 0..in_ch {
                    let w_base_local = (oc * in_ch + ic) * 9;
                    let w_base_grad = weight_offset + w_base_local;

                    for oy in 0..bs {
                        for ox in 0..bs {
                            let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                            let d = d_pre[out_idx];
                            if d == 0.0 {
                                continue;
                            }

                            for ky in 0..3i32 {
                                for kx in 0..3i32 {
                                    let iy = oy + ky - 1;
                                    let ix = ox + kx - 1;
                                    if iy >= 0 && iy < bs && ix >= 0 && ix < bs {
                                        let in_idx = ic * spatial + (iy as usize) * bs as usize + ix as usize;
                                        let k_idx = (ky * 3 + kx) as usize;
                                        // Weight gradient
                                        grad[w_base_grad + k_idx] += d * input[in_idx];
                                        // Input gradient
                                        d_input[in_idx] += d * layer_weights[w_base_local + k_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        KernelType::Diamond => {
            let mask = diamond_mask_5x5();
            for oc in 0..out_ch {
                for ic in 0..in_ch {
                    let w_base_local = (oc * in_ch + ic) * 25;
                    let w_base_grad = weight_offset + w_base_local;

                    for oy in 0..bs {
                        for ox in 0..bs {
                            let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                            let d = d_pre[out_idx];
                            if d == 0.0 {
                                continue;
                            }

                            for ky in 0..5i32 {
                                for kx in 0..5i32 {
                                    let k_idx = (ky * 5 + kx) as usize;
                                    if !mask[k_idx] {
                                        continue;
                                    }
                                    let iy = oy + ky - 2;
                                    let ix = ox + kx - 2;
                                    if iy >= 0 && iy < bs && ix >= 0 && ix < bs {
                                        let in_idx = ic * spatial + (iy as usize) * bs as usize + ix as usize;
                                        grad[w_base_grad + k_idx] += d * input[in_idx];
                                        d_input[in_idx] += d * layer_weights[w_base_local + k_idx];
                                    }
                                }
                            }
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

            for oc in 0..out_ch {
                for ic in 0..in_ch {
                    let bw_base = (oc * in_ch + ic) * 9;
                    let hw_base = (oc * in_ch + ic) * 5;
                    let vw_base = (oc * in_ch + ic) * 5;

                    for oy in 0..bs {
                        for ox in 0..bs {
                            let out_idx = oc * spatial + (oy as usize) * bs as usize + ox as usize;
                            let d = d_pre[out_idx];
                            if d == 0.0 {
                                continue;
                            }

                            // Box 3x3
                            for ky in 0..3i32 {
                                for kx in 0..3i32 {
                                    let iy = oy + ky - 1;
                                    let ix = ox + kx - 1;
                                    if iy >= 0 && iy < bs && ix >= 0 && ix < bs {
                                        let in_idx = ic * spatial + (iy as usize) * bs as usize + ix as usize;
                                        let k_idx = (ky * 3 + kx) as usize;
                                        grad[box_grad_off + bw_base + k_idx] += d * input[in_idx];
                                        d_input[in_idx] += d * box_w[bw_base + k_idx];
                                    }
                                }
                            }

                            // Horizontal 1x5
                            {
                                let iy = oy;
                                if iy >= 0 && iy < bs {
                                    for kx in 0..5i32 {
                                        let ix = ox + kx - 2;
                                        if ix >= 0 && ix < bs {
                                            let in_idx = ic * spatial + (iy as usize) * bs as usize + ix as usize;
                                            grad[h_grad_off + hw_base + kx as usize] += d * input[in_idx];
                                            d_input[in_idx] += d * h_w[hw_base + kx as usize];
                                        }
                                    }
                                }
                            }

                            // Vertical 5x1
                            for ky in 0..5i32 {
                                let iy = oy + ky - 2;
                                let ix = ox;
                                if iy >= 0 && iy < bs && ix >= 0 && ix < bs {
                                    let in_idx = ic * spatial + (iy as usize) * bs as usize + ix as usize;
                                    grad[v_grad_off + vw_base + ky as usize] += d * input[in_idx];
                                    d_input[in_idx] += d * v_w[vw_base + ky as usize];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── CnnEvaluator (for game playing) ──────────────────────────────────────

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

/// Apply diamond mask: zero out corner weights in all diamond conv layers.
/// Call after each optimizer step to enforce the mask.
pub fn apply_diamond_mask(model: &mut CnnModel) {
    if model.kernel_type != KernelType::Diamond {
        return;
    }
    let mask = diamond_mask_5x5();
    let mut offset = 0;
    let mut in_ch = model.input_channels;
    for &out_ch in &model.conv_channels {
        let n_w = out_ch * in_ch * 25;
        for i in 0..n_w {
            let kpos = i % 25;
            if !mask[kpos] {
                model.weights[offset + i] = 0.0;
            }
        }
        offset += n_w + out_ch; // weights + bias
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
}
