use std::sync::Arc;
use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{NdArray, Wgpu};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use duke_rust::game::ai::heuristics::{HeuristicAi, Heuristics};
use duke_rust::game::ai::player::{AiMove, ArtificialPlayer, EvaluatingPlayer};
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::bag::TileBag;
use duke_rust::game::board_setup::{DukeInitialLocation, FootmenSetup};
use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::tile::Owner;
use duke_rust::game::units;

use duke_training::encoding::encode_state;
use duke_training::model::ValueNetwork;

#[derive(Clone, Copy)]
enum BackendKind {
    Cpu,
    Gpu,
}

fn create_bag() -> TileBag {
    TileBag::new(vec![
        Arc::new(units::footman()),
        Arc::new(units::bowman()),
        Arc::new(units::knight()),
        Arc::new(units::pikeman()),
        Arc::new(units::pikeman()),
        Arc::new(units::champion()),
        Arc::new(units::priest()),
        Arc::new(units::wizard()),
        Arc::new(units::dragoon()),
        Arc::new(units::general()),
        Arc::new(units::marshall()),
        Arc::new(units::longbowman()),
    ])
}

fn create_initial_state(bag: &TileBag) -> GameState {
    GameState::new(
        bag,
        (DukeInitialLocation::Left, FootmenSetup::Left),
        (DukeInitialLocation::Right, FootmenSetup::Right),
    )
}

/// Neural network greedy player: picks the move that maximizes V(s').
fn nn_greedy_move<B: Backend, R: Rng>(
    gs: &GameState,
    model: &ValueNetwork<B>,
    device: &B::Device,
    rng: &mut R,
) -> AiMove {
    let mut moves: Vec<AiMove> = AiMove::all_moves(gs).collect();
    moves.shuffle(rng);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_move = None;

    for mv in &moves {
        let mut clone = gs.clone();
        mv.play(&mut clone, rng);
        let encoded = encode_state::<B>(&clone, device);
        let batch = encoded.unsqueeze::<4>();
        let prediction: f32 = model.forward(batch).into_scalar().elem();
        let score = 1.0 - prediction as f64;
        if score > best_score {
            best_score = score;
            best_move = Some(mv.clone());
        }
    }

    best_move.unwrap()
}

/// Heuristic greedy player: picks the move with highest cheap_evaluate.
fn heuristic_greedy_move<R: Rng>(
    gs: &GameState,
    evaluator: &dyn EvaluatingPlayer,
    rng: &mut R,
) -> AiMove {
    let mut moves: Vec<AiMove> = AiMove::all_moves(gs).collect();
    moves.shuffle(rng);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_move = None;

    for mv in &moves {
        let mut clone = gs.clone();
        mv.play(&mut clone, rng);
        let score = -evaluator.cheap_evaluate(&clone);
        if score > best_score {
            best_score = score;
            best_move = Some(mv.clone());
        }
    }

    best_move.unwrap()
}

enum Player<'a, B: Backend> {
    Random,
    Heuristic(&'a dyn EvaluatingPlayer),
    NeuralNet(&'a ValueNetwork<B>, &'a B::Device),
}

fn play_match<B: Backend>(
    gs: &GameState,
    top_player: &Player<B>,
    bottom_player: &Player<B>,
    rng: &mut StdRng,
    max_turns: u32,
) -> GameResult {
    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut turns = 0u32;

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                if turns >= max_turns {
                    return GameResult::Tie;
                }
                let current = game.current_player_turn();
                let player = match current {
                    Owner::TopPlayer => top_player,
                    Owner::BottomPlayer => bottom_player,
                };
                match player {
                    Player::Random => { ai.play_next_move(rng, &mut game); }
                    Player::Heuristic(eval) => {
                        let mv = heuristic_greedy_move(&game, *eval, rng);
                        mv.play(&mut game, rng);
                    }
                    Player::NeuralNet(model, device) => {
                        let mv = nn_greedy_move(&game, model, *device, rng);
                        mv.play(&mut game, rng);
                    }
                }
                turns += 1;
            }
            result => return result,
        }
    }
}

struct MatchResult {
    player_a_wins: u32,
    player_b_wins: u32,
    ties: u32,
}

fn run_matches<B: Backend>(
    gs: &GameState,
    player_a: &Player<B>,
    player_b: &Player<B>,
    num_games: u32,
    label: &str,
) -> MatchResult {
    let start = Instant::now();
    let mut result = MatchResult { player_a_wins: 0, player_b_wins: 0, ties: 0 };

    for seed in 0..num_games {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let game_result = if seed % 2 == 0 {
            let r = play_match(gs, player_a, player_b, &mut rng, 200);
            match r {
                GameResult::Won(Owner::TopPlayer) => GameResult::Won(Owner::TopPlayer),
                GameResult::Won(Owner::BottomPlayer) => GameResult::Won(Owner::BottomPlayer),
                other => other,
            }
        } else {
            let r = play_match(gs, player_b, player_a, &mut rng, 200);
            match r {
                GameResult::Won(Owner::TopPlayer) => GameResult::Won(Owner::BottomPlayer),
                GameResult::Won(Owner::BottomPlayer) => GameResult::Won(Owner::TopPlayer),
                other => other,
            }
        };
        match game_result {
            GameResult::Won(Owner::TopPlayer) => result.player_a_wins += 1,
            GameResult::Won(Owner::BottomPlayer) => result.player_b_wins += 1,
            _ => result.ties += 1,
        }
    }

    let elapsed = start.elapsed();
    let total = num_games as f64;
    println!(
        "{}: A={:.1}% B={:.1}% Tie={:.1}% ({} games in {:.1?})",
        label,
        result.player_a_wins as f64 / total * 100.0,
        result.player_b_wins as f64 / total * 100.0,
        result.ties as f64 / total * 100.0,
        num_games,
        elapsed,
    );

    result
}

fn run_all_benchmarks<B: Backend>(device: B::Device, checkpoint: &str, backend_name: &str) {
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    println!("Loading model from: {} (backend: {})", checkpoint, backend_name);
    let model = ValueNetwork::<B>::new(&device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model
        .load_file(checkpoint, &recorder, &device)
        .expect("Failed to load model checkpoint");
    println!("Model loaded.\n");

    let evaluator = HeuristicAi::new(vec![
        Box::new(Heuristics::DukeMovementOptions),
        Box::new(Heuristics::TotalTilesOnBoard),
        Box::new(Heuristics::TotalMovementOptions),
        Box::new(Heuristics::DiscardedUnits),
    ]);

    let nn = Player::NeuralNet(&model, &device);
    let random = Player::Random;
    let heuristic = Player::Heuristic(&evaluator);

    let num_games = 200;

    println!("=== Benchmark: {} games per matchup ({}) ===\n", num_games, backend_name);

    println!("--- Neural Net vs Random ---");
    run_matches(&gs, &nn, &random, num_games, "NN vs Random");

    println!("\n--- Neural Net vs Heuristic (greedy depth-1) ---");
    run_matches(&gs, &nn, &heuristic, num_games, "NN vs Heuristic");

    println!("\n--- Heuristic vs Random (baseline) ---");
    run_matches(&gs, &heuristic, &random, num_games, "Heuristic vs Random");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: benchmark <checkpoint_path> [--gpu|--cpu]");
        eprintln!("  e.g.: benchmark checkpoints/model_game_100000 --cpu");
        eprintln!("  Default: --cpu");
        std::process::exit(1);
    }

    let checkpoint = &args[1];
    let backend = if args.iter().any(|a| a == "--gpu") {
        BackendKind::Gpu
    } else {
        BackendKind::Cpu
    };

    match backend {
        BackendKind::Cpu => run_all_benchmarks::<NdArray>(Default::default(), checkpoint, "CPU"),
        BackendKind::Gpu => run_all_benchmarks::<Wgpu>(WgpuDevice::default(), checkpoint, "GPU"),
    }
}
