//! CLI tool for the model registry.
//!
//! Usage:
//!   model_registry [--db <path>] list
//!   model_registry [--db <path>] register <model_path> [--description "..."]
//!   model_registry [--db <path>] info <id>
//!   model_registry [--db <path>] benchmarks <id>
//!
//! Default DB path: D:/temp/duke_models.db

use duke_training::generic_mlp::GenericMlp;
use duke_training::learned_heuristic::load_lr_weights_raw;
use duke_training::model_registry::{ModelRecord, ModelRegistry};
use duke_training::nnue::{NnueWeights, NUM_FEATURES};

const DEFAULT_DB: &str = "D:/temp/duke_models.db";

fn usage() -> ! {
    eprintln!("Usage: model_registry [--db <path>] <command> [args]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  list                          List all registered models");
    eprintln!("  register <path> [--description \"...\"]  Register a model file");
    eprintln!("  info <id>                     Show model details + benchmarks");
    eprintln!("  benchmarks <id>               Show benchmark history");
    eprintln!();
    eprintln!("Default DB: {}", DEFAULT_DB);
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Parse --db flag
    let mut db_path = DEFAULT_DB.to_string();
    let mut remaining: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--db" {
            i += 1;
            if i >= args.len() {
                eprintln!("Error: --db requires a value");
                std::process::exit(1);
            }
            db_path = args[i].clone();
        } else {
            remaining.push(args[i].clone());
        }
        i += 1;
    }

    if remaining.is_empty() {
        usage();
    }

    let command = &remaining[0];
    match command.as_str() {
        "list" => cmd_list(&db_path),
        "register" => cmd_register(&db_path, &remaining[1..]),
        "info" => cmd_info(&db_path, &remaining[1..]),
        "benchmarks" => cmd_benchmarks(&db_path, &remaining[1..]),
        other => {
            eprintln!("Unknown command: {}", other);
            usage();
        }
    }
}

fn cmd_list(db_path: &str) {
    let reg = ModelRegistry::open(db_path).expect("Failed to open registry DB");
    let models = reg.list_models().expect("Failed to list models");

    if models.is_empty() {
        println!("No models registered.");
        return;
    }

    println!(
        "{:>4}  {:>6}  {:>22}  {:>8}  {:>8}  {}",
        "ID", "Format", "Architecture", "Params", "Created", "Path"
    );
    println!("{}", "-".repeat(90));

    for m in &models {
        let created_short = if m.created_at.len() >= 10 {
            &m.created_at[..10]
        } else {
            &m.created_at
        };
        println!(
            "{:>4}  {:>6}  {:>22}  {:>8}  {:>8}  {}",
            m.id, m.file_format, m.architecture, m.param_count, created_short, m.file_path,
        );
    }
    println!("\n{} model(s) total.", models.len());
}

fn cmd_register(db_path: &str, args: &[String]) {
    if args.is_empty() {
        eprintln!("Error: register requires a model file path");
        usage();
    }

    let model_path = &args[0];

    // Parse optional --description
    let mut description: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--description" {
            i += 1;
            if i >= args.len() {
                eprintln!("Error: --description requires a value");
                std::process::exit(1);
            }
            description = Some(args[i].clone());
        }
        i += 1;
    }

    let reg = ModelRegistry::open(db_path).expect("Failed to open registry DB");

    let id = if model_path.ends_with(".gmlp") {
        let net = GenericMlp::load(model_path).expect("Failed to load .gmlp file");
        println!(
            "Loaded GMLP: {} ({} params)",
            net.arch_string(),
            net.weights.len()
        );
        reg.register_gmlp(model_path, &net, description.as_deref(), None)
            .expect("Failed to register model")
    } else if model_path.ends_with(".nnue") {
        let weights = NnueWeights::load(model_path).expect("Failed to load .nnue file");
        let param_count = weights.l1_weight.len()
            + weights.l1_bias.len()
            + weights.l2_weight.len()
            + weights.l2_bias.len()
            + weights.l3_weight.len()
            + weights.l3_bias.len();
        println!(
            "Loaded NNUE: {}->{}->{}->1 ({} params)",
            NUM_FEATURES, weights.l1_size, weights.l2_size, param_count
        );
        reg.register_nnue(model_path, &weights, description.as_deref(), None)
            .expect("Failed to register model")
    } else if model_path.ends_with(".json") {
        let raw = load_lr_weights_raw(model_path).expect("Failed to load .json weight file");
        let n = raw.len();
        let label = match n {
            24 => "LR-Guard",
            41 => "LR-Cheap",
            65 => "LR-All",
            _ => {
                eprintln!(
                    "JSON weight file has {} weights. Expected 24 (LR-Guard), 41 (LR-Cheap), or 65 (LR-All).",
                    n
                );
                std::process::exit(1);
            }
        };
        println!("Loaded {}: {} weights", label, n);
        reg.register_lr(model_path, n, description.as_deref(), None)
            .expect("Failed to register model")
    } else {
        eprintln!(
            "Unknown file format: {}. Expected .gmlp, .nnue, or .json extension.",
            model_path
        );
        std::process::exit(1);
    };

    println!("Registered as model ID {}.", id);
}

fn cmd_info(db_path: &str, args: &[String]) {
    if args.is_empty() {
        eprintln!("Error: info requires a model ID");
        usage();
    }

    let id: i64 = args[0]
        .parse()
        .expect("Model ID must be a positive integer");

    let reg = ModelRegistry::open(db_path).expect("Failed to open registry DB");

    match reg.get_model(id).expect("Failed to query model") {
        None => {
            eprintln!("Model ID {} not found.", id);
            std::process::exit(1);
        }
        Some(m) => {
            print_model_detail(&m);

            let benchmarks = reg.get_benchmarks(id).expect("Failed to get benchmarks");
            if benchmarks.is_empty() {
                println!("\nNo benchmarks recorded.");
            } else {
                println!("\nBenchmarks ({}):", benchmarks.len());
                print_benchmarks(&benchmarks);
            }
        }
    }
}

fn cmd_benchmarks(db_path: &str, args: &[String]) {
    if args.is_empty() {
        eprintln!("Error: benchmarks requires a model ID");
        usage();
    }

    let id: i64 = args[0]
        .parse()
        .expect("Model ID must be a positive integer");

    let reg = ModelRegistry::open(db_path).expect("Failed to open registry DB");

    // Verify the model exists before querying benchmarks
    if reg.get_model(id).expect("Failed to query model").is_none() {
        eprintln!("Model #{} not found.", id);
        std::process::exit(1);
    }

    let benchmarks = reg.get_benchmarks(id).expect("Failed to get benchmarks");

    if benchmarks.is_empty() {
        println!("No benchmarks recorded for model ID {}.", id);
        return;
    }

    println!("Benchmarks for model ID {} ({} records):", id, benchmarks.len());
    print_benchmarks(&benchmarks);
}

fn print_model_detail(m: &ModelRecord) {
    println!("Model ID: {}", m.id);
    println!("  Path:         {}", m.file_path);
    println!("  Format:       {}", m.file_format);
    println!("  Architecture: {}", m.architecture);
    println!("  Input size:   {}", m.input_size);
    println!("  Param count:  {}", m.param_count);
    if let Some(ref desc) = m.description {
        println!("  Description:  {}", desc);
    }
    println!("  Created:      {}", m.created_at);
    if let Some(iters) = m.training_iterations {
        println!("  Training iterations: {}", iters);
    }
    if let Some(sigma) = m.training_sigma {
        println!("  Training sigma:      {:.4}", sigma);
    }
    if let Some(lr) = m.training_lr {
        println!("  Training LR:         {:.4}", lr);
    }
    if let Some(ref opp) = m.training_opponent {
        println!("  Training opponent:   {}", opp);
    }
    if let Some(parent) = m.parent_model_id {
        println!("  Parent model ID:     {}", parent);
    }
}

fn print_benchmarks(benchmarks: &[duke_training::model_registry::BenchmarkRecord]) {
    println!(
        "  {:>10}  {:>6}  {:>5}  {:>5}  {:>5}  {:>7}  {:>6}  {}",
        "Opponent", "Games", "Wins", "Loss", "Ties", "WinRate", "Elo", "Date"
    );
    println!("  {}", "-".repeat(75));
    for b in benchmarks {
        let opp_short = if b.opponent.len() > 10 {
            &b.opponent[b.opponent.len() - 10..]
        } else {
            &b.opponent
        };
        let date_short = b
            .benchmark_date
            .as_ref()
            .map(|d| {
                if d.len() >= 10 {
                    &d[..10]
                } else {
                    d.as_str()
                }
            })
            .unwrap_or("?");
        println!(
            "  {:>10}  {:>6}  {:>5}  {:>5}  {:>5}  {:>6.1}%  {:>6}  {}",
            opp_short,
            b.num_games,
            b.wins,
            b.losses,
            b.ties,
            b.win_rate.unwrap_or(0.0) * 100.0,
            b.elo
                .map(|e| format!("{:.0}", e))
                .unwrap_or_else(|| "-".to_string()),
            date_short,
        );
    }
}
