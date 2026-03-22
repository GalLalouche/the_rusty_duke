use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use duke_rust::game::state::GameResult;

use duke_training::fc_td_training::{FcTdTrainer, GameTrajectory};
use duke_training::game_setup::{
    create_bag, create_initial_state, play_random_game, play_selfplay_game,
    GameEvaluator, StaticHeuristicEvaluator,
};
use duke_training::learned_heuristic::RegressionAccumulator;
use duke_training::nnue::NnueEvaluator;
use duke_training::trajectory_io::TrajectoryWriter;
use duke_training::weight_export::export_weights;

type MyBackend = Autodiff<Wgpu>;

/// How games are played during training.
#[derive(Clone, Copy, PartialEq)]
pub enum PlayMode {
    /// Pure random moves (phase 1 baseline).
    Random,
    /// 1-ply heuristic greedy with epsilon-greedy exploration.
    Heuristic,
    /// NNUE self-play with epsilon-greedy exploration.
    SelfPlay,
}

/// All knobs for a training run, parsed from command-line arguments.
pub struct TrainingConfig {
    pub total_games: u64,
    pub lr_start: f64,
    pub lr_end: f64,
    pub epsilon: f64,
    pub update_interval: u64,
    pub play_mode: PlayMode,
    pub resume_path: Option<String>,
    pub checkpoint_dir: String,
    pub l1_size: usize,
    pub l2_size: usize,
    pub batch_size: u64,
}

impl TrainingConfig {
    /// Linear decay from lr_start to lr_end over total_games.
    pub fn lr_at(&self, game_num: u64) -> f64 {
        if self.total_games == 0 { return self.lr_start; }
        let progress = game_num as f64 / self.total_games as f64;
        self.lr_start + (self.lr_end - self.lr_start) * progress
    }
}

/// Parse a typed flag value from CLI args: `--flag <value>`.
fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

/// Parse a string flag value from CLI args: `--flag <value>`.
fn parse_flag_string(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

impl TrainingConfig {
    pub fn from_args(args: &[String]) -> Self {
        let play_mode = if args.iter().any(|a| a == "--self-play") {
            PlayMode::SelfPlay
        } else if args.iter().any(|a| a == "--heuristic") {
            PlayMode::Heuristic
        } else {
            PlayMode::Random
        };

        let total_games: u64 = parse_flag(args, "--games").unwrap_or(100_000);
        let lr_start: f64 = parse_flag(args, "--lr").unwrap_or(0.1);
        let lr_end: f64 = parse_flag(args, "--lr-end").unwrap_or(lr_start / 100.0);
        let epsilon: f64 = parse_flag(args, "--epsilon").unwrap_or(0.15);
        let update_interval: u64 = parse_flag(args, "--update-interval").unwrap_or(1000);
        let resume_path: Option<String> = parse_flag_string(args, "--resume");
        let checkpoint_dir: String = parse_flag_string(args, "--checkpoint-dir")
            .unwrap_or_else(|| "checkpoints".to_string());
        let l1_size: usize = parse_flag(args, "--l1").unwrap_or(256);
        let l2_size: usize = parse_flag(args, "--l2").unwrap_or(32);

        let default_batch_size = std::thread::available_parallelism()
            .map(|n| n.get() as u64)
            .unwrap_or(4);
        let batch_size: u64 = parse_flag(args, "--batch-size").unwrap_or(default_batch_size).max(1);

        Self {
            total_games,
            lr_start,
            lr_end,
            epsilon,
            update_interval,
            play_mode,
            resume_path,
            checkpoint_dir,
            l1_size,
            l2_size,
            batch_size,
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
    let heuristic_evaluator = StaticHeuristicEvaluator::new();

    match config.play_mode {
        PlayMode::SelfPlay => {
            println!("NNUE self-play training");
            println!(
                "  epsilon={}, update_interval={}, total_games={}, lr={}->{}, batch_size={}",
                config.epsilon, config.update_interval, config.total_games,
                config.lr_start, config.lr_end, config.batch_size
            );
            if config.resume_path.is_none() {
                eprintln!("WARNING: --self-play without --resume starts from random weights!");
            }
        }
        PlayMode::Heuristic => {
            println!("Heuristic 1-ply training");
            println!(
                "  epsilon={}, total_games={}, lr={}->{}, batch_size={}",
                config.epsilon, config.total_games,
                config.lr_start, config.lr_end, config.batch_size
            );
        }
        PlayMode::Random => {
            println!("Random play training");
            println!("  total_games={}, lr={}->{}, batch_size={}",
                config.total_games, config.lr_start, config.lr_end, config.batch_size);
        }
    }

    // Accumulate data for learned heuristic regression (free — no extra games needed)
    let mut regression_acc = RegressionAccumulator::new();

    // Save game trajectories to disk for offline replay
    let trajectory_path = format!("{}/trajectories.dtrj", config.checkpoint_dir);
    std::fs::create_dir_all(&config.checkpoint_dir).expect("Failed to create checkpoints dir");
    let mut traj_writer = TrajectoryWriter::new(&trajectory_path)
        .expect("Failed to create trajectory file");
    println!("Saving trajectories to: {}", trajectory_path);

    let start = Instant::now();
    let mut total_loss = 0.0f32;
    let mut recent_loss = 0.0f32;
    let mut wins = [0u32; 2];
    let mut ties = 0u32;
    let mut game_num: u64 = 0;
    let mut recent_game_count: u64 = 0;
    let mut last_logged_at: u64 = 0;

    while game_num < config.total_games {
        let batch_size = config.batch_size.min(config.total_games - game_num);

        // Update learning rate at round boundary
        trainer.set_lr(config.lr_at(game_num));

        // Play batch_size games in parallel (CPU, rayon)
        let evaluator: &(dyn GameEvaluator + Sync) = match config.play_mode {
            PlayMode::SelfPlay => &nnue_evaluator,
            PlayMode::Heuristic => &heuristic_evaluator,
            PlayMode::Random => &heuristic_evaluator, // unused, but needed for type
        };
        let trajectories: Vec<GameTrajectory> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let seed = game_num + i;
                let mut rng = StdRng::seed_from_u64(seed);
                let (states, result) = if config.play_mode == PlayMode::Random {
                    play_random_game(&gs, &mut rng)
                } else {
                    play_selfplay_game(&gs, evaluator, &mut rng, config.epsilon)
                };
                GameTrajectory { states, result }
            })
            .collect();

        // Phase 2: Batch-train on all trajectories (single forward+backward pass)
        let loss = trainer.train_on_batch(&trajectories);
        total_loss += loss * batch_size as f32;
        recent_loss += loss * batch_size as f32;

        // Accumulate for learned heuristic regression + save to disk
        for traj in &trajectories {
            regression_acc.add_game(&traj.states, &traj.result);
            traj_writer.write_game(&traj.states, &traj.result)
                .expect("Failed to write trajectory");
        }

        // Update stats
        for traj in &trajectories {
            match traj.result {
                GameResult::Won(duke_rust::game::tile::Owner::TopPlayer) => wins[0] += 1,
                GameResult::Won(duke_rust::game::tile::Owner::BottomPlayer) => wins[1] += 1,
                GameResult::Tie => ties += 1,
                _ => {}
            }
        }

        game_num += batch_size;
        recent_game_count += batch_size;

        // Log every 100 games (or at round boundaries that cross the threshold)
        if game_num / 100 > last_logged_at / 100 || game_num >= config.total_games {
            let avg_loss = total_loss / game_num as f32;
            let recent_avg = if recent_game_count > 0 {
                recent_loss / recent_game_count as f32
            } else {
                0.0
            };
            let elapsed = start.elapsed();
            println!(
                "Game {}: avg_loss={:.6}, recent_loss={:.6}, lr={:.6}, wins=[{}, {}], ties={}, elapsed={:.1?}",
                game_num, avg_loss, recent_avg, trainer.lr(), wins[0], wins[1], ties, elapsed
            );
            recent_loss = 0.0;
            recent_game_count = 0;
            last_logged_at = game_num;
        }

        // Checkpoint at update_interval boundaries
        if game_num / config.update_interval > (game_num - batch_size) / config.update_interval {
            let checkpoint_game = (game_num / config.update_interval) * config.update_interval;
            let checkpoint_path = format!("{}/fc_model_game_{}", config.checkpoint_dir, checkpoint_game);
            std::fs::create_dir_all(&config.checkpoint_dir).expect("Failed to create checkpoints dir");
            trainer.save_model(&checkpoint_path);

            let nnue_path = format!("{}/nnue_game_{}.nnue", config.checkpoint_dir, checkpoint_game);
            let new_weights = export_weights(&trainer.model, config.l1_size, config.l2_size);
            new_weights.save(&nnue_path).expect("Failed to save NNUE weights");

            // Sync trajectory file so it's recoverable on crash
            traj_writer.sync().expect("Failed to sync trajectory file");

            if config.play_mode == PlayMode::SelfPlay {
                nnue_evaluator = NnueEvaluator::new(new_weights);
                println!("NNUE weights updated + checkpoint saved: {}", nnue_path);
            } else {
                println!("Checkpoint saved: {}", nnue_path);
            }
        }
    }

    // Finalize trajectory file
    let traj_count = traj_writer.finish().expect("Failed to finalize trajectory file");

    let elapsed = start.elapsed();
    println!("\nTraining complete: {} games in {:.1?}", config.total_games, elapsed);
    if config.total_games > 0 {
        println!("Avg time per game: {:.1?}", elapsed.div_f64(config.total_games as f64));
        println!("Final avg loss: {:.6}", total_loss as f64 / config.total_games as f64);
    }

    // Save regression accumulator (X'X + X'y) for instant lambda sweeps later
    let reg_path = format!("{}/regression.bin", config.checkpoint_dir);
    regression_acc.save(&reg_path).expect("Failed to save regression accumulator");
    println!("Regression accumulator saved: {} ({} samples)", reg_path, regression_acc.n_samples());

    // Solve and save learned heuristic weights (free regression from the same games)
    let learned_weights = regression_acc.solve();
    let lr_path = format!("{}/learned_heuristic.json", config.checkpoint_dir);
    learned_weights.save(&lr_path).expect("Failed to save learned heuristic weights");
    println!("Learned heuristic saved: {} ({} samples)", lr_path, regression_acc.n_samples());
    println!("Trajectories saved: {} ({} games)", trajectory_path, traj_count);
}
