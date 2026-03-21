use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

use crate::encoding::TOTAL_FEATURES;
use crate::nnue;

pub const INPUT_SIZE: usize = TOTAL_FEATURES; // 1106 (1080 board + 26 bag)
pub const L1_SIZE: usize = nnue::L1_SIZE;
pub const L2_SIZE: usize = nnue::L2_SIZE;

/// A flat fully-connected value network matching the NNUE architecture.
///
/// Architecture:
///   Input: [batch, 1080]
///   -> Linear(1080 -> 256) + ReLU
///   -> Linear(256 -> 32) + ReLU
///   -> Linear(32 -> 1) + Sigmoid
///
/// Output: single value in [0, 1] representing win probability for the
/// current player.
#[derive(Module, Debug)]
pub struct FcValueNetwork<B: Backend> {
    pub fc1: Linear<B>,  // 1080 -> 256
    pub fc2: Linear<B>,  // 256 -> 32
    pub fc3: Linear<B>,  // 32 -> 1
}

impl<B: Backend> FcValueNetwork<B> {
    /// Create a new FC value network with randomly initialized weights.
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(INPUT_SIZE, L1_SIZE).init(device),
            fc2: LinearConfig::new(L1_SIZE, L2_SIZE).init(device),
            fc3: LinearConfig::new(L2_SIZE, 1).init(device),
        }
    }

    /// Forward pass. Input shape: [batch, 1080]. Output shape: [batch, 1].
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.fc2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.fc3.forward(x);
        sigmoid(x)
    }
}
