use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

use crate::encoding::TOTAL_FEATURES;

pub const INPUT_SIZE: usize = TOTAL_FEATURES;

#[derive(Module, Debug)]
pub struct FcValueNetwork<B: Backend> {
    pub layers: Vec<Linear<B>>,
}

impl<B: Backend> FcValueNetwork<B> {
    /// Create a new fully-connected value network.
    ///
    /// `hidden_sizes` specifies the size of each hidden layer. For example:
    /// - `&[256, 32]` creates INPUT_SIZE -> 256 -> 32 -> 1
    /// - `&[128]` creates INPUT_SIZE -> 128 -> 1
    pub fn new(device: &B::Device, hidden_sizes: &[usize]) -> Self {
        assert!(!hidden_sizes.is_empty(), "Need at least one hidden layer");
        let mut layers = Vec::new();
        let mut prev = INPUT_SIZE;
        for &h in hidden_sizes {
            assert!(h > 0, "Hidden layer size must be > 0");
            layers.push(LinearConfig::new(prev, h).init(device));
            prev = h;
        }
        // Output layer
        layers.push(LinearConfig::new(prev, 1).init(device));
        Self { layers }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < self.layers.len() - 1 {
                // Hidden layers: ReLU
                x = burn::tensor::activation::relu(x);
            }
        }
        // Output: sigmoid
        sigmoid(x)
    }

    /// Returns the number of layers (hidden + output).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}
