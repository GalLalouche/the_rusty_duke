use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use rand::rngs::StdRng;
use rand::SeedableRng;

use duke_rust::game::state::GameResult;

use duke_training::fc_td_training::FcTdTrainer;
use duke_training::game_setup::{create_bag, create_initial_state, play_random_game, play_selfplay_game};
use duke_training::nnue::NnueEvaluator;
use duke_training::weight_export::export_weights;

type MyBackend = Autodiff<Wgpu>;

/// All knobs for a training run, parsed from command-line arguments.
pub struct TrainingConfig {
    pub total_games: u64,
    pub lr_start: f64,
    pub lr_end: f64,
    pub epsilon: f64,
    pub update_interval: u64,
    pub self_play: bool,
    pub resume_path: Option<String>,
    pub checkpoint_dir: String,
    pub l1_size: usize,
    pub l2_size: usize,
}

impl TrainingConfig {
    /// Linear decay from lr_start to lr_end over total_games.
    pub fn lr_at(&self, game_num: u64) -> f64 {
        if self.total_games == 0 { return self.lr_start; }
        let progress = game_num as f64 / self.total_games as f64;
        self.lr_start + (self.lr_end - self.lr_start) * progress
    }
}

impl TrainingConfig {
    pub fn from_args(args: &[String]) -> Self {
        let self_play = args.iter().any(|a| a == "--self-play");

        let total_games: u64 = args
            .iter()
            .position(|a| a == "--games")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(100_000);

        let lr_start: f64 = args
            .iter()
            .position(|a| a == "--lr")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.1);

        let lr_end: f64 = args
            .iter()
            .position(|a| a == "--lr-end")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(lr_start / 100.0);

        let epsilon: f64 = args
            .iter()
            .position(|a| a == "--epsilon")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.15);

        let update_interval: u64 = args
            .iter()
            .position(|a| a == "--update-interval")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);

        let resume_path: Option<String> = args
            .iter()
            .position(|a| a == "--resume")
            .and_then(|i| args.get(i + 1))
            .cloned();

        let checkpoint_dir: String = args
            .iter()
            .position(|a| a == "--checkpoint-dir")
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| "checkpoints".to_string());

        let l1_size: usize = args
            .iter()
            .position(|a| a == "--l1")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);

        let l2_size: usize = args
            .iter()
            .position(|a| a == "--l2")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);

        Self {
            total_games,
            lr_start,
            lr_end,
            epsilon,
            update_interval,
            self_play,
            resume_path,
            checkpoint_dir,
            l1_size,
            l2_size,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = TrainingConfig::from_args(&args);

    let bag = create_bag();
    let gs = create_initial_state(&bag);

    let device = WgpuDevice::default();
    let mut trainer: FcTdTrainer<MyBackend> = FcTdTrainer::new(device, config.lr_start, config.l1_size, config.l2_size);
    println!("  network: {}→{}→{}→1", duke_training::encoding::TOTAL_FEATURES, config.l1_size, config.l2_size);

    // Load checkpoint if resuming
    if let Some(ref path) = config.resume_path {
        println!("Resuming from checkpoint: {}", path);
        trainer.load_model(path);
        println!("Model loaded.");
    }

    // Initialize NNUE evaluator from the (possibly loaded) model
    let nnue_weights = export_weights(&trainer.model, config.l1_size, config.l2_size);
    let mut nnue_evaluator = NnueEvaluator::new(nnue_weights);

    if config.self_play {
        println!("Phase 2: NNUE self-play training");
        println!(
            "  epsilon={}, update_interval={}, total_games={}, lr={}->{}",
            config.epsilon, config.update_interval, config.total_games,
            config.lr_start, config.lr_end
        );
        if config.resume_path.is_none() {
            eprintln!("WARNING: --self-play without --resume starts from random weights!");
        }
    } else {
        println!("Phase 1: Random play training");
        println!("  total_games={}, lr={}->{}",
            config.total_games, config.lr_start, config.lr_end);
    }

    let start = Instant::now();
    let mut total_loss = 0.0f32;
    let mut recent_loss = 0.0f32;
    let mut wins = [0u32; 2];
    let mut ties = 0u32;

    for game_num in 0..config.total_games {
        // Update learning rate per schedule
        trainer.set_lr(config.lr_at(game_num));

        let mut rng = StdRng::seed_from_u64(game_num);

        let (states, result) = if config.self_play {
            play_selfplay_game(&gs, &nnue_evaluator, &mut rng, config.epsilon)
        } else {
            play_random_game(&gs, &mut rng)
        };

        let loss = trainer.train_on_game(&states, result);
        total_loss += loss;
        recent_loss += loss;

        match result {
            GameResult::Won(duke_rust::game::tile::Owner::TopPlayer) => wins[0] += 1,
            GameResult::Won(duke_rust::game::tile::Owner::BottomPlayer) => wins[1] += 1,
            GameResult::Tie => ties += 1,
            _ => {}
        }

        if (game_num + 1) % 100 == 0 {
            let avg_loss = total_loss / (game_num + 1) as f32;
            let recent_avg = recent_loss / 100.0;
            let elapsed = start.elapsed();
            println!(
                "Game {}: avg_loss={:.6}, recent_loss={:.6}, lr={:.6}, wins=[{}, {}], ties={}, elapsed={:.1?}",
                game_num + 1, avg_loss, recent_avg, trainer.lr(), wins[0], wins[1], ties, elapsed
            );
            recent_loss = 0.0;
        }

        if (game_num + 1) % config.update_interval == 0 {
            // Save checkpoints
            let checkpoint_path = format!("{}/fc_model_game_{}", config.checkpoint_dir, game_num + 1);
            std::fs::create_dir_all(&config.checkpoint_dir).expect("Failed to create checkpoints dir");
            trainer.save_model(&checkpoint_path);

            let nnue_path = format!("{}/nnue_game_{}.nnue", config.checkpoint_dir, game_num + 1);
            let new_weights = export_weights(&trainer.model, config.l1_size, config.l2_size);
            new_weights.save(&nnue_path).expect("Failed to save NNUE weights");

            if config.self_play {
                // Update NNUE evaluator with latest trained weights
                nnue_evaluator = NnueEvaluator::new(new_weights);
                println!("NNUE weights updated + checkpoint saved: {}", nnue_path);
            } else {
                println!("Checkpoint saved: {}", nnue_path);
            }
        }
    }

    let elapsed = start.elapsed();
    println!("\nTraining complete: {} games in {:.1?}", config.total_games, elapsed);
    println!("Avg time per game: {:.1?}", elapsed.div_f64(config.total_games as f64));
    println!("Final avg loss: {:.6}", total_loss as f64 / config.total_games as f64);
}
