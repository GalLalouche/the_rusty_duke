//! Train NNUE from saved game trajectories (no game replay needed).
//!
//! Usage: train_from_trajectories --trajectories <path.dtrj> --l1 <size> --l2 <size>
//!          [--lr 0.1] [--lr-end 0.001] [--epochs 1] [--batch-size 24]
//!          [--checkpoint-dir <dir>]

use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};

use duke_training::cli::parse_flag;
use duke_training::fc_td_training::{FcTdTrainer, GameTrajectory};
use duke_training::trajectory_io::load_trajectories;
use duke_training::weight_export::export_weights;

type MyBackend = Autodiff<Wgpu>;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let traj_path = parse_flag::<String>(&args, "--trajectories")
        .expect("Usage: train_from_trajectories --trajectories <path.dtrj> --l1 <size> --l2 <size>");

    let l1_size: usize = parse_flag(&args, "--l1").unwrap_or(256);
    let l2_size: usize = parse_flag(&args, "--l2").unwrap_or(32);
    let lr_start: f64 = parse_flag(&args, "--lr").unwrap_or(0.1);
    let lr_end: f64 = parse_flag(&args, "--lr-end").unwrap_or(lr_start / 100.0);
    let epochs: usize = parse_flag(&args, "--epochs").unwrap_or(1);
    let batch_size: usize = parse_flag(&args, "--batch-size").unwrap_or(24);
    let checkpoint_dir = parse_flag::<String>(&args, "--checkpoint-dir")
        .unwrap_or_else(|| "checkpoints_offline".to_string());

    println!("Loading trajectories from: {}", traj_path);
    let t = Instant::now();
    let raw_games = load_trajectories(&traj_path).expect("Failed to load trajectories");
    let total_states: usize = raw_games.iter().map(|g| g.states.len()).sum();
    println!("  {} games, {} states in {:.1?}", raw_games.len(), total_states, t.elapsed());

    // Convert to GameTrajectory format
    let games: Vec<GameTrajectory> = raw_games.into_iter()
        .map(|g| GameTrajectory { states: g.states, result: g.result })
        .collect();

    println!("Training NNUE: {}→{}→{}→1", duke_training::encoding::TOTAL_FEATURES, l1_size, l2_size);
    println!("  lr={}->{}, epochs={}, batch_size={}", lr_start, lr_end, epochs, batch_size);

    let device = WgpuDevice::default();
    let mut trainer: FcTdTrainer<MyBackend> = FcTdTrainer::new(device, lr_start, l1_size, l2_size);

    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint dir");

    let total_batches = (games.len() + batch_size - 1) / batch_size;
    let total_steps = total_batches * epochs;

    let start = Instant::now();
    let mut step = 0usize;
    let mut total_loss = 0.0f32;

    for epoch in 0..epochs {
        for batch_start in (0..games.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(games.len());
            let batch = &games[batch_start..batch_end];

            // Linear LR decay across all steps
            let progress = step as f64 / total_steps.max(1) as f64;
            let lr = lr_start + (lr_end - lr_start) * progress;
            trainer.set_lr(lr);

            let loss = trainer.train_on_batch(batch);
            total_loss += loss;
            step += 1;

            if step % 100 == 0 || step == total_steps {
                let avg_loss = total_loss / step as f32;
                println!(
                    "  epoch {}/{}, batch {}/{}, step {}/{}: loss={:.6}, avg_loss={:.6}, lr={:.6}, {:.1?}",
                    epoch + 1, epochs, batch_start / batch_size + 1, total_batches,
                    step, total_steps, loss, avg_loss, lr, start.elapsed()
                );
            }
        }

        // Save checkpoint at end of each epoch (both burn model + NNUE weights)
        let model_path = format!("{}/model_epoch_{}", checkpoint_dir, epoch + 1);
        trainer.save_model(&model_path);
        let nnue_path = format!("{}/nnue_epoch_{}.nnue", checkpoint_dir, epoch + 1);
        let weights = export_weights(&trainer.model, l1_size, l2_size);
        weights.save(&nnue_path).expect("Failed to save NNUE weights");
        println!("  Checkpoint saved: {} + {}", model_path, nnue_path);
    }

    let elapsed = start.elapsed();
    println!("\nTraining complete in {:.1?}", elapsed);
    println!("Final avg loss: {:.6}", total_loss / step as f32);
}
