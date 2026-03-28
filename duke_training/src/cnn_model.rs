//! CNN-based value network for The Duke.
//!
//! Input: 30 board planes (6x6) + 26 bag features.
//! Conv layers maintain 6x6 spatial dimensions via padding, with optional
//! diamond (manhattan-2) kernel masking for 5x5 convolutions.

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

use crate::encoding::{BAG_FEATURES, BOARD_SIZE, NUM_BOARD_PLANES};

/// CNN value network: conv layers over board planes, concatenated with bag
/// features, then FC layers to a scalar sigmoid output.
#[derive(Module, Debug)]
pub struct CnnValueNetwork<B: Backend> {
    conv_layers: Vec<Conv2d<B>>,
    fc_layers: Vec<Linear<B>>,
    /// Flattened conv output size (last_channels * 6 * 6) + bag features (26).
    fc_input_size: usize,
}

impl<B: Backend> CnnValueNetwork<B> {
    /// Create a new CNN value network.
    ///
    /// - `conv_channels`: channel count for each conv layer, e.g. `[64, 64, 32]`.
    ///   The first conv layer takes `NUM_BOARD_PLANES` (30) input channels.
    /// - `fc_sizes`: hidden FC layer sizes, e.g. `[128]`.
    /// - `kernel_size`: spatial kernel size (3 for box, 5 for diamond).
    pub fn new(
        device: &B::Device,
        conv_channels: &[usize],
        fc_sizes: &[usize],
        kernel_size: usize,
    ) -> Self {
        assert!(!conv_channels.is_empty(), "Need at least one conv layer");
        assert!(kernel_size == 3 || kernel_size == 5,
            "kernel_size must be 3 (box) or 5 (diamond), got {}", kernel_size);

        let padding = (kernel_size - 1) / 2;

        let mut conv_layers = Vec::new();
        let mut in_channels = NUM_BOARD_PLANES;
        for &out_channels in conv_channels {
            let config = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
                .with_padding(PaddingConfig2d::Explicit(padding, padding));
            conv_layers.push(config.init(device));
            in_channels = out_channels;
        }

        let conv_flat_size = conv_channels.last().unwrap() * BOARD_SIZE * BOARD_SIZE;
        let fc_input_size = conv_flat_size + BAG_FEATURES;

        let mut fc_layers = Vec::new();
        let mut prev = fc_input_size;
        for &h in fc_sizes {
            fc_layers.push(LinearConfig::new(prev, h).init(device));
            prev = h;
        }
        // Output layer: single scalar
        fc_layers.push(LinearConfig::new(prev, 1).init(device));

        Self {
            conv_layers,
            fc_layers,
            fc_input_size,
        }
    }

    /// Forward pass.
    ///
    /// - `board`: `[batch, 30, 6, 6]` float tensor (board planes).
    /// - `bag`: `[batch, 26]` float tensor (bag features).
    ///
    /// Returns: `[batch, 1]` sigmoid output.
    pub fn forward(&self, board: Tensor<B, 4>, bag: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = board;
        for (i, conv) in self.conv_layers.iter().enumerate() {
            x = conv.forward(x);
            // ReLU between conv layers (and after the last conv, before flatten)
            x = burn::tensor::activation::relu(x);
            let _ = i; // suppress unused warning
        }

        // Flatten conv output: [batch, channels, 6, 6] -> [batch, channels * 36]
        let [batch_size, channels, h, w] = x.dims();
        let flat_size = channels * h * w;
        let x_flat = x.reshape([batch_size, flat_size]);

        // Concatenate with bag features
        let combined = Tensor::cat(vec![x_flat, bag], 1); // [batch, conv_flat + 26]

        // FC layers
        let mut x = combined;
        for (i, layer) in self.fc_layers.iter().enumerate() {
            x = layer.forward(x);
            if i < self.fc_layers.len() - 1 {
                x = burn::tensor::activation::relu(x);
            }
        }

        // Output: sigmoid
        sigmoid(x)
    }

    /// Return the total fc_input_size (for diagnostics).
    pub fn fc_input_size(&self) -> usize {
        self.fc_input_size
    }
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
///
/// Returns a flat `[5 * 5]` mask with 1.0 for valid positions and 0.0 for corners.
pub fn diamond_mask_5x5() -> [f32; 25] {
    let mut mask = [0.0f32; 25];
    for dy in 0..5i32 {
        for dx in 0..5i32 {
            let dist = (dy - 2).unsigned_abs() + (dx - 2).unsigned_abs();
            if dist <= 2 {
                mask[(dy * 5 + dx) as usize] = 1.0;
            }
        }
    }
    mask
}

/// Apply the diamond mask to all conv layers that use 5x5 kernels.
///
/// For each conv layer whose kernel is 5x5, multiplies the weight tensor
/// element-wise by the diamond mask (broadcast over out_channels and in_channels).
/// This zeroes out the four corner positions in each 5x5 kernel.
///
/// Call this after each optimizer step when using `--kernel diamond`.
pub fn apply_diamond_mask<B: Backend>(model: &mut CnnValueNetwork<B>, device: &B::Device) {
    let mask_flat = diamond_mask_5x5();
    for conv in model.conv_layers.iter_mut() {
        let w = conv.weight.val();
        let [out_c, in_c, kh, kw] = w.dims();
        if kh == 5 && kw == 5 {
            // Build mask tensor [1, 1, 5, 5] and broadcast
            let mask_tensor = Tensor::<B, 4>::from_floats(
                burn::tensor::TensorData::new(mask_flat.to_vec(), [1, 1, 5, 5]),
                device,
            );
            // Expand to [out_c, in_c, 5, 5]
            let mask_expanded = mask_tensor.expand([out_c, in_c, 5, 5]);
            let masked_w = w.mul(mask_expanded);
            conv.weight = burn::module::Param::from_tensor(masked_w);
        }
    }
}
