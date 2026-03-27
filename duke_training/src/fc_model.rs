use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

use crate::encoding::TOTAL_FEATURES;

pub const INPUT_SIZE: usize = TOTAL_FEATURES;

#[derive(Module, Debug)]
pub struct FcValueNetwork<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
    pub fc3: Linear<B>,
}

impl<B: Backend> FcValueNetwork<B> {
    pub fn new(device: &B::Device, l1_size: usize, l2_size: usize) -> Self {
        assert!(l1_size > 0, "l1_size must be > 0");
        assert!(l2_size > 0, "l2_size must be > 0");
        Self {
            fc1: LinearConfig::new(INPUT_SIZE, l1_size).init(device),
            fc2: LinearConfig::new(l1_size, l2_size).init(device),
            fc3: LinearConfig::new(l2_size, 1).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.fc2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.fc3.forward(x);
        sigmoid(x)
    }
}
