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
use duke_training::nnue::NnueEvaluator;

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

/// Timing accumulators for profiling NN move selection.
struct NnTimings {
    move_gen: std::time::Duration,
    play_encode: std::time::Duration,
    stack: std::time::Duration,
    forward: std::time::Duration,
    extract: std::time::Duration,
    calls: u64,
}

impl NnTimings {
    fn new() -> Self {
        Self {
            move_gen: std::time::Duration::ZERO,
            play_encode: std::time::Duration::ZERO,
            stack: std::time::Duration::ZERO,
            forward: std::time::Duration::ZERO,
            extract: std::time::Duration::ZERO,
            calls: 0,
        }
    }
    fn print(&self) {
        if self.calls == 0 { return; }
        let total = self.move_gen + self.play_encode + self.stack + self.forward + self.extract;
        println!("\n  NN move profiling ({} calls, {:.1?} total, {:.2?}/call):",
            self.calls, total, total / self.calls as u32);
        println!("    move_gen:     {:.1?} ({:.0}%)", self.move_gen, self.move_gen.as_secs_f64() / total.as_secs_f64() * 100.0);
        println!("    play+encode:  {:.1?} ({:.0}%)", self.play_encode, self.play_encode.as_secs_f64() / total.as_secs_f64() * 100.0);
        println!("    tensor stack: {:.1?} ({:.0}%)", self.stack, self.stack.as_secs_f64() / total.as_secs_f64() * 100.0);
        println!("    forward pass: {:.1?} ({:.0}%)", self.forward, self.forward.as_secs_f64() / total.as_secs_f64() * 100.0);
        println!("    extract:      {:.1?} ({:.0}%)", self.extract, self.extract.as_secs_f64() / total.as_secs_f64() * 100.0);
    }
}

/// Neural network greedy player: picks the move that maximizes V(s').
fn nn_greedy_move<B: Backend, R: Rng>(
    gs: &GameState,
    model: &ValueNetwork<B>,
    device: &B::Device,
    rng: &mut R,
    timings: &mut NnTimings,
) -> AiMove {
    let t0 = Instant::now();
    let mut moves: Vec<AiMove> = AiMove::all_moves(gs).collect();
    moves.shuffle(rng);
    timings.move_gen += t0.elapsed();

    let t1 = Instant::now();
    let encoded: Vec<Tensor<B, 3>> = moves
        .iter()
        .map(|mv| {
            let mut clone = gs.clone();
            mv.play(&mut clone, rng);
            encode_state::<B>(&clone, device)
        })
        .collect();
    timings.play_encode += t1.elapsed();

    let t2 = Instant::now();
    let batch = Tensor::stack(encoded, 0);
    timings.stack += t2.elapsed();

    let t3 = Instant::now();
    let predictions = model.forward(batch);
    timings.forward += t3.elapsed();

    let t4 = Instant::now();
    let scores: Vec<f32> = predictions
        .squeeze::<1>(1)
        .into_data()
        .to_vec()
        .expect("to_vec");
    timings.extract += t4.elapsed();

    timings.calls += 1;

    let best_idx = scores
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    moves.swap_remove(best_idx)
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
        let mut eval_rng = StdRng::seed_from_u64(0);
        mv.play(&mut clone, &mut eval_rng);
        let score = -evaluator.cheap_evaluate(&clone);
        if score > best_score {
            best_score = score;
            best_move = Some(mv.clone());
        }
    }

    best_move.unwrap()
}

/// NNUE greedy player: picks the move that maximizes value after the move.
fn nnue_greedy_move<R: Rng>(
    gs: &GameState,
    evaluator: &NnueEvaluator,
    rng: &mut R,
) -> AiMove {
    let mut moves: Vec<AiMove> = AiMove::all_moves(gs).collect();
    moves.shuffle(rng);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_move = None;

    for mv in &moves {
        let mut clone = gs.clone();
        // Use a deterministic rng for play so candidate evaluation doesn't
        // corrupt the main rng or depend on move order.
        let mut eval_rng = StdRng::seed_from_u64(0);
        mv.play(&mut clone, &mut eval_rng);
        let prediction = evaluator.evaluate_state(&clone);
        let score = 1.0 - prediction as f64;
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
    Nnue(&'a NnueEvaluator),
}

fn play_match<B: Backend>(
    gs: &GameState,
    top_player: &Player<B>,
    bottom_player: &Player<B>,
    rng: &mut StdRng,
    max_turns: u32,
    timings: &mut NnTimings,
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
                        let mv = nn_greedy_move(&game, model, *device, rng, timings);
                        mv.play(&mut game, rng);
                    }
                    Player::Nnue(evaluator) => {
                        let mv = nnue_greedy_move(&game, evaluator, rng);
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
    let mut timings = NnTimings::new();

    for seed in 0..num_games {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let game_result = if seed % 2 == 0 {
            let r = play_match(gs, player_a, player_b, &mut rng, 200, &mut timings);
            match r {
                GameResult::Won(Owner::TopPlayer) => GameResult::Won(Owner::TopPlayer),
                GameResult::Won(Owner::BottomPlayer) => GameResult::Won(Owner::BottomPlayer),
                other => other,
            }
        } else {
            let r = play_match(gs, player_b, player_a, &mut rng, 200, &mut timings);
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
    timings.print();

    result
}

fn run_all_benchmarks<B: Backend>(
    device: B::Device,
    checkpoint: Option<&str>,
    backend_name: &str,
    nnue_path: Option<&str>,
) {
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    let model = if let Some(cp) = checkpoint {
        println!("Loading CNN model from: {} (backend: {})", cp, backend_name);
        let m = ValueNetwork::<B>::new(&device);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let m = m.load_file(cp, &recorder, &device)
            .expect("Failed to load model checkpoint");
        println!("CNN model loaded.\n");
        Some(m)
    } else {
        println!("No CNN checkpoint provided, skipping NN benchmarks.\n");
        None
    };

    let evaluator = HeuristicAi::new(vec![
        Box::new(Heuristics::DukeMovementOptions),
        Box::new(Heuristics::TotalTilesOnBoard),
        Box::new(Heuristics::TotalMovementOptions),
        Box::new(Heuristics::DiscardedUnits),
    ]);

    let nnue_evaluator = nnue_path.map(|path| {
        println!("Loading NNUE weights from: {}", path);
        let weights = duke_training::nnue::NnueWeights::load(path)
            .expect("Failed to load NNUE weights");
        let eval = NnueEvaluator::new(weights);
        println!("NNUE weights loaded.\n");
        eval
    });

    let random: Player<B> = Player::Random;
    let heuristic: Player<B> = Player::Heuristic(&evaluator);

    let num_games = std::env::var("NUM_GAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200u32);

    println!("=== Benchmark: {} games per matchup ({}) ===\n", num_games, backend_name);

    if let Some(ref m) = model {
        let nn = Player::NeuralNet(m, &device);
        println!("--- Neural Net vs Random ---");
        run_matches(&gs, &nn, &random, num_games, "NN vs Random");

        println!("\n--- Neural Net vs Heuristic (greedy depth-1) ---");
        run_matches(&gs, &nn, &heuristic, num_games, "NN vs Heuristic");
    }

    println!("\n--- Heuristic vs Random (baseline) ---");
    run_matches(&gs, &heuristic, &random, num_games, "Heuristic vs Random");

    if let Some(ref nnue_eval) = nnue_evaluator {
        let nnue_player = Player::<B>::Nnue(nnue_eval);

        println!("\n--- NNUE vs Random ---");
        run_matches(&gs, &nnue_player, &random, num_games, "NNUE vs Random");

        println!("\n--- NNUE vs Heuristic ---");
        run_matches(&gs, &nnue_player, &heuristic, num_games, "NNUE vs Heuristic");

        if let Some(ref m) = model {
            let nn = Player::NeuralNet(m, &device);
            println!("\n--- NNUE vs Neural Net ---");
            run_matches(&gs, &nnue_player, &nn, num_games, "NNUE vs NN");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // First positional arg is CNN checkpoint (optional if --nnue is provided)
    let checkpoint = args.get(1)
        .filter(|a| !a.starts_with("--"))
        .map(|s| s.as_str());
    let backend = if args.iter().any(|a| a == "--gpu") {
        BackendKind::Gpu
    } else {
        BackendKind::Cpu
    };

    let nnue_path = args
        .iter()
        .position(|a| a == "--nnue")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    if checkpoint.is_none() && nnue_path.is_none() {
        eprintln!("Usage: benchmark [checkpoint_path] [--gpu|--cpu] [--nnue <weights_path>]");
        eprintln!("  e.g.: benchmark checkpoints/model_game_100000 --cpu");
        eprintln!("  e.g.: benchmark --nnue model.nnue");
        std::process::exit(1);
    }

    match backend {
        BackendKind::Cpu => {
            run_all_benchmarks::<NdArray>(Default::default(), checkpoint, "CPU", nnue_path)
        }
        BackendKind::Gpu => {
            run_all_benchmarks::<Wgpu>(WgpuDevice::default(), checkpoint, "GPU", nnue_path)
        }
    }
}
