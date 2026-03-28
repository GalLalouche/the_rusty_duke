use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{GradientsParams, Optimizer, SgdConfig, Sgd};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;

use duke_rust::game::state::{GameResult, GameState};

use crate::encoding::encode_state_flat;
use crate::fc_model::FcValueNetwork;
// Re-export GameTrajectory from trajectory_io for backward compatibility.
pub use crate::trajectory_io::GameTrajectory;

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
    ///
    /// `hidden_sizes` specifies the size of each hidden layer. For example:
    /// - `&[256, 32]` creates INPUT_SIZE -> 256 -> 32 -> 1
    /// - `&[128]` creates INPUT_SIZE -> 128 -> 1
    pub fn new(device: B::Device, lr: f64, hidden_sizes: &[usize]) -> Self {
        let model = FcValueNetwork::new(&device, hidden_sizes);
        let optimizer = SgdConfig::new().init::<B, FcValueNetwork<B>>();
        Self {
            model,
            optimizer,
            lr,
            device,
        }
    }

    /// Train on a batch of completed game trajectories in a single forward+backward pass.
    ///
    /// Flattens all states from all games, computes TD targets respecting game
    /// boundaries (each game's terminal state uses the actual outcome, not the
    /// next game's first state), then performs one combined gradient step.
    ///
    /// Returns the average loss across all states.
    pub fn train_on_batch(&mut self, games: &[GameTrajectory]) -> f32 {
        // Flatten all states and record game boundary information
        let mut all_states: Vec<&GameState> = Vec::new();
        // For each state, record whether it's the last in its game, and if so, the result
        let mut game_boundary: Vec<Option<GameResult>> = Vec::new();

        for game in games {
            if game.states.len() < 2 {
                continue;
            }
            let n = game.states.len();
            for (i, state) in game.states.iter().enumerate() {
                all_states.push(state);
                if i == n - 1 {
                    game_boundary.push(Some(game.result));
                } else {
                    game_boundary.push(None);
                }
            }
        }

        if all_states.len() < 2 {
            return 0.0;
        }

        let total_states = all_states.len();

        // Encode all states as a batch and run forward pass
        let tensors: Vec<Tensor<B, 1>> = all_states
            .iter()
            .map(|gs| encode_state_flat::<B>(gs, &self.device))
            .collect();
        let batch = Tensor::stack(tensors, 0); // [total_states, TOTAL_FEATURES]
        let predictions = self.model.forward(batch); // [total_states, 1]
        let predictions = predictions.squeeze::<1>(1); // [total_states]

        // Detach predictions for building targets (no gradient through targets)
        let pred_data: Vec<f32> = predictions
            .clone()
            .into_data()
            .to_vec()
            .expect("Failed to convert predictions to vec");

        // Build TD targets respecting game boundaries
        let mut targets = Vec::with_capacity(total_states);
        for t in 0..total_states {
            if let Some(result) = game_boundary[t] {
                // Terminal state: use actual outcome for the current player
                let current_player = all_states[t].current_player_turn();
                let outcome = match result {
                    GameResult::Won(winner) => {
                        if winner == current_player { 1.0f32 } else { 0.0f32 }
                    }
                    GameResult::Tie => 0.5f32,
                    GameResult::Ongoing => unreachable!("Game should be finished"),
                };
                targets.push(outcome);
            } else {
                // Non-terminal: TD target is 1 - V(s_{t+1}) (opponent's perspective)
                debug_assert!(t + 1 < total_states,
                    "Non-terminal state at index {} but total_states is {} (game boundary logic bug)",
                    t, total_states);
                targets.push(1.0 - pred_data[t + 1]);
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

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Apply gradients with the current learning rate. For use by external training loops.
    pub fn optimizer_step(&mut self, grads: GradientsParams) -> FcValueNetwork<B> {
        self.optimizer.step(self.lr, self.model.clone(), grads)
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
