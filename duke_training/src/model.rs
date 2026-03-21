use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

use crate::encoding::{BOARD_SIZE, NUM_PLANES};

/// A small CNN value network for board evaluation.
///
/// Architecture:
///   Input: [batch, 30, 6, 6]
///   -> Conv2d(30 -> 64, 3x3, pad=1) + ReLU
///   -> Conv2d(64 -> 64, 3x3, pad=1) + ReLU
///   -> Flatten to [batch, 64*6*6]
///   -> Linear(64*6*6 -> 128) + ReLU
///   -> Linear(128 -> 1) + Sigmoid
///
/// Output: single value in [0, 1] representing win probability for the
/// current player.
#[derive(Module, Debug)]
pub struct ValueNetwork<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

const CONV_CHANNELS: usize = 64;
const FC_HIDDEN: usize = 128;
const FLAT_SIZE: usize = CONV_CHANNELS * BOARD_SIZE * BOARD_SIZE;

impl<B: Backend> ValueNetwork<B> {
    /// Create a new value network with randomly initialized weights.
    pub fn new(device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([NUM_PLANES, CONV_CHANNELS], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv2 = Conv2dConfig::new([CONV_CHANNELS, CONV_CHANNELS], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let fc1 = LinearConfig::new(FLAT_SIZE, FC_HIDDEN).init(device);
        let fc2 = LinearConfig::new(FC_HIDDEN, 1).init(device);

        Self {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }

    /// Forward pass. Input shape: `[batch, 30, 6, 6]`. Output shape: `[batch, 1]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let batch_size = x.dims()[0];

        // Conv layers with ReLU
        let x = self.conv1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.conv2.forward(x);
        let x = burn::tensor::activation::relu(x);

        // Flatten spatial dimensions
        let x = x.reshape([batch_size as i32, FLAT_SIZE as i32]);

        // Fully connected layers
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.fc2.forward(x);

        sigmoid(x)
    }
}
