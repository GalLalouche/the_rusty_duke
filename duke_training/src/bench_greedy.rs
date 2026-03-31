use std::time::Instant;
use rand::rngs::StdRng;
use rand::SeedableRng;
use duke_rust::game::state::GameResult;
use duke_training::game_setup::{
    create_bag, create_initial_state, greedy_move, GameEvaluator, StaticHeuristicEvaluator,
};
use duke_training::loaded_model::LoadedModel;

const MAX_TURNS: u32 = 500;

fn bench(eval: &dyn GameEvaluator, label: &str, num_games: u32) {
    let bag = create_bag();
    let init_state = create_initial_state(&bag);

    let mut total_moves = 0u64;
    let start = Instant::now();

    for seed in 0..num_games {
        let mut game = init_state.clone();
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut turns = 0u32;
        loop {
            if game.game_result() != GameResult::Ongoing || turns >= MAX_TURNS {
                break;
            }
            turns += 1;
            let mv = greedy_move(&mut game, eval, &mut rng);
            mv.play(&mut game, &mut rng);
        }
        total_moves += turns as u64;
    }

    let elapsed = start.elapsed();
    println!("[{}] Games: {}, Moves: {}, Time: {:.3}s",
        label, num_games, total_moves, elapsed.as_secs_f64());
    println!("  {:.1} us/move, {:.1} ms/game",
        elapsed.as_secs_f64() * 1_000_000.0 / total_moves as f64,
        elapsed.as_secs_f64() * 1_000.0 / num_games as f64);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_games: u32 = duke_training::cli::parse_flag(&args, "--games").unwrap_or(2000);

    if args.contains(&"--synthetic-fc".to_string()) {
        let hidden_str: String = duke_training::cli::parse_flag(&args, "--hidden")
            .unwrap_or_else(|| "256,256,256,256,256,256".to_string());
        let hidden: Vec<usize> = hidden_str.split(',')
            .map(|s| s.trim().parse().unwrap()).collect();
        let mut rng = StdRng::seed_from_u64(42);
        let net = duke_training::generic_mlp::GenericMlp::random(
            duke_training::encoding::TOTAL_FEATURES, hidden.clone(), &mut rng);
        let eval = duke_training::generic_mlp::GenericEvaluator { net };
        let label = format!("FC {}", hidden.iter().map(|h| h.to_string()).collect::<Vec<_>>().join("x"));
        bench(&eval, &label, num_games);
        return;
    }

    if args.contains(&"--synthetic-cnn".to_string()) {
        let conv_str: String = duke_training::cli::parse_flag(&args, "--conv")
            .unwrap_or_else(|| "64,64,32".to_string());
        let conv_channels: Vec<usize> = conv_str.split(',')
            .map(|s| s.trim().parse().unwrap()).collect();
        let fc_str: String = duke_training::cli::parse_flag(&args, "--fc")
            .unwrap_or_else(|| "128".to_string());
        let fc_sizes: Vec<usize> = fc_str.split(',')
            .map(|s| s.trim().parse().unwrap()).collect();
        let kernel_str: String = duke_training::cli::parse_flag(&args, "--kernel")
            .unwrap_or_else(|| "box".to_string());
        let kernel = match kernel_str.as_str() {
            "box" => duke_training::cnn::KernelType::Box,
            "diamond" => duke_training::cnn::KernelType::Diamond,
            "cross" => duke_training::cnn::KernelType::Cross,
            other => panic!("Unknown kernel: {}", other),
        };
        let mut rng = StdRng::seed_from_u64(42);
        let model = duke_training::cnn::CnnModel::random(
            kernel,
            duke_training::encoding::NUM_BOARD_PLANES,
            conv_channels.clone(), fc_sizes.clone(),
            duke_training::encoding::BOARD_SIZE,
            duke_training::encoding::BAG_FEATURES,
            &mut rng,
        );
        let eval = duke_training::cnn::CnnEvaluator { model };
        let label = format!("CNN {:?} conv={:?} fc={:?}", kernel, conv_channels, fc_sizes);
        bench(&eval, &label, num_games);
        return;
    }

    if args.len() > 1 && !args[1].starts_with("--") {
        let spec = &args[1];
        let quantize = args.contains(&"--quantize".to_string());
        let model = LoadedModel::from_spec(spec, quantize);
        if let Some(ref eval) = model.evaluator {
            bench(eval.as_ref(), &model.label, num_games);
        } else {
            eprintln!("Cannot benchmark Random player");
        }
    } else {
        let eval = StaticHeuristicEvaluator::new();
        bench(&eval, "StaticHeuristic", num_games);
    }
}
