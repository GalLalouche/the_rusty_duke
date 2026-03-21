use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{GradientsParams, Optimizer, SgdConfig, Sgd};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;

use duke_rust::game::state::{GameResult, GameState};

use crate::encoding::encode_state_flat;
use crate::fc_model::FcValueNetwork;

/// TD trainer that plays games and trains the FC value network using temporal
/// difference learning. Same approach as TdTrainer but using the flat FC model.
pub struct FcTdTrainer<B: AutodiffBackend> {
    pub model: FcValueNetwork<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, FcValueNetwork<B>, B>,
    lr: f64,
    device: B::Device,
}

impl<B: AutodiffBackend> FcTdTrainer<B> {
    /// Create a new trainer with a fresh model and SGD optimizer.
    pub fn new(device: B::Device, lr: f64, l1_size: usize, l2_size: usize) -> Self {
        let model = FcValueNetwork::new(&device, l1_size, l2_size);
        let optimizer = SgdConfig::new().init::<B, FcValueNetwork<B>>();
        Self {
            model,
            optimizer,
            lr,
            device,
        }
    }

    /// Encode a list of game states into a batched tensor of shape
    /// `[batch, 1080]`.
    fn encode_batch(&self, states: &[GameState]) -> Tensor<B, 2> {
        let tensors: Vec<Tensor<B, 1>> = states
            .iter()
            .map(|gs| encode_state_flat::<B>(gs, &self.device))
            .collect();
        Tensor::stack(tensors, 0)
    }

    /// Train on a single completed game trajectory.
    ///
    /// `states` - the sequence of game states observed during the game
    ///            (one per turn, before the move is applied).
    /// `result` - the final game result (must not be `Ongoing`).
    ///
    /// The encoding is always from the current player's perspective, so
    /// successive states alternate perspective. The TD target for state t is
    /// `1 - V(s_{t+1})` (opponent's value flipped), or the actual outcome
    /// for the terminal state.
    ///
    /// Returns the average loss for this game.
    pub fn train_on_game(
        &mut self,
        states: &[GameState],
        result: GameResult,
    ) -> f32 {
        if states.len() < 2 {
            return 0.0;
        }

        let n = states.len();

        // Encode all states as a batch and run forward pass
        let batch = self.encode_batch(states);
        let predictions = self.model.forward(batch); // [n, 1]
        let predictions = predictions.squeeze::<1>(1); // [n]

        // Detach predictions for building targets (no gradient through targets)
        let pred_data: Vec<f32> = predictions
            .clone()
            .into_data()
            .to_vec()
            .expect("Failed to convert predictions to vec");

        // Build TD targets
        let mut targets = Vec::with_capacity(n);
        for t in 0..n {
            if t < n - 1 {
                // Next state is from opponent's perspective, so flip the value
                targets.push(1.0 - pred_data[t + 1]);
            } else {
                // Terminal state: use actual outcome for the current player
                let current_player = states[t].current_player_turn();
                let outcome = match result {
                    GameResult::Won(winner) => {
                        if winner == current_player {
                            1.0f32
                        } else {
                            0.0f32
                        }
                    }
                    GameResult::Tie => 0.5f32,
                    GameResult::Ongoing => unreachable!("Game should be finished"),
                };
                targets.push(outcome);
            }
        }

        let target_tensor =
            Tensor::<B, 1>::from_floats(targets.as_slice(), &self.device);

        // MSE loss: mean((predictions - targets)^2)
        let diff = predictions - target_tensor;
        let loss = diff.clone().mul(diff).mean();

        let loss_value: f32 = loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("loss")[0];

        // Backward pass and optimizer step
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optimizer.step(self.lr, self.model.clone(), grads);

        loss_value
    }

    /// Update the learning rate (e.g., for decay schedules).
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    pub fn lr(&self) -> f64 {
        self.lr
    }

    /// Save model weights to a file.
    ///
    /// The file extension (`.mpk`) is automatically appended by the recorder.
    pub fn save_model(&self, path: &str) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.model
            .clone()
            .save_file(path, &recorder)
            .expect("Failed to save model");
    }

    /// Load model weights from a file and reset the optimizer state.
    ///
    /// The file extension (`.mpk`) is automatically appended by the recorder.
    pub fn load_model(&mut self, path: &str) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.model = self
            .model
            .clone()
            .load_file(path, &recorder, &self.device)
            .expect("Failed to load model");
        // Reset optimizer state to match the loaded model
        self.optimizer = SgdConfig::new().init::<B, FcValueNetwork<B>>();
    }
}
