#!/usr/bin/env python3
"""
Generic overnight ES training runner.

Loads experiment definitions from a TOML config file and runs them
sequentially via es_train. Each experiment gets its own checkpoint
subdirectory and a top-level log file in the output directory.

Usage:
    python overnight_run.py experiments.toml
"""

import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# TOML parsing: Python 3.11+ has tomllib, older versions need tomli
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        print("ERROR: TOML support required. Install tomli: pip install tomli")
        sys.exit(1)


def find_es_train():
    """Auto-detect es_train binary relative to this script's location.

    Script lives in duke_training/scripts/, binary is at
    ../../target/release/es_train(.exe).
    """
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent  # duke_training/scripts -> duke_training -> repo root

    if platform.system() == "Windows":
        binary = repo_root / "target" / "release" / "es_train.exe"
    else:
        binary = repo_root / "target" / "release" / "es_train"

    return binary, repo_root


def load_config(config_path):
    """Load and validate a TOML experiment config file."""
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    # Validate required sections
    if "common" not in cfg:
        print("ERROR: Config must have a [common] section")
        sys.exit(1)
    if "experiments" not in cfg or not cfg["experiments"]:
        print("ERROR: Config must have at least one [[experiments]] entry")
        sys.exit(1)

    return cfg


def build_common_args(common):
    """Build the common CLI args list from the [common] config section."""
    args = []
    flag_map = {
        "pop": "--pop",
        "games": "--games",
        "sigma": "--sigma",
        "lr": "--lr",
        "eval_interval": "--eval-interval",
        "eval_games": "--eval-games",
        "iterations": "--iterations",
        "time_limit": "--time-limit",
    }
    for key, flag in flag_map.items():
        if key in common:
            args.extend([flag, str(common[key])])
    return args


def resolve_output_dir(common):
    """Determine the output directory from config or generate a timestamped one."""
    if "output_dir" in common and common["output_dir"]:
        return Path(common["output_dir"])
    date_str = datetime.now().strftime("%Y-%m-%d")
    return Path(f"D:/temp/overnight_{date_str}")


def run_experiment(exp, index, total, es_train_exe, repo_root, common_args, output_dir):
    """Run a single experiment, capturing output to a log file."""
    name = exp["name"]
    exp_dir = output_dir / exp.get("dir", name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Log files go at the top level of output_dir so monitor.py can find them
    log_path = output_dir / f"{name}.log"

    extra_args = exp.get("args", [])
    cmd = [str(es_train_exe)] + common_args + extra_args + [
        "--checkpoint-dir", str(exp_dir),
    ]

    print(f"\n{'='*70}")
    print(f"  Experiment {index}/{total}: {name}")
    print(f"  Checkpoint: {exp_dir}")
    print(f"  Command:    {' '.join(cmd)}")
    print(f"  Log:        {log_path}")
    print(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    sys.stdout.flush()

    start = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(repo_root),
        )
    elapsed = time.time() - start

    status = "OK" if proc.returncode == 0 else f"FAILED (exit={proc.returncode})"
    print(f"  Result: {status} in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    sys.stdout.flush()

    return {
        "name": name,
        "dir": str(exp_dir),
        "log": str(log_path),
        "exit_code": proc.returncode,
        "elapsed_s": elapsed,
    }


def collect_eval_lines(log_path):
    """Extract all EVAL lines from a log file."""
    evals = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                if "EVAL:" in line:
                    evals.append(line.strip())
    except FileNotFoundError:
        pass
    return evals


def parse_eval_winrate(line):
    """Parse an EVAL line to extract win rate A%. Returns float or None."""
    m = re.search(r"A=([\d.]+)%", line)
    if m:
        return float(m.group(1))
    return None


def write_summary(results, output_dir):
    """Write a summary comparison table to the output directory."""
    summary_path = output_dir / "summary.txt"
    date_str = datetime.now().strftime("%Y-%m-%d")

    lines = []
    lines.append(f"Overnight Run Summary - {date_str}")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"{'Experiment':<30} {'Exit':>5} {'Time':>8} {'Best%':>7} {'Last%':>7} {'Evals':>6}")
    lines.append(f"{'-'*30} {'-'*5} {'-'*8} {'-'*7} {'-'*7} {'-'*6}")

    for r in results:
        evals = collect_eval_lines(r["log"])
        winrates = [parse_eval_winrate(e) for e in evals]
        winrates = [w for w in winrates if w is not None]

        best_wr = f"{max(winrates):.1f}%" if winrates else "N/A"
        last_wr = f"{winrates[-1]:.1f}%" if winrates else "N/A"
        n_evals = str(len(evals))
        status = "OK" if r["exit_code"] == 0 else f"ERR{r['exit_code']}"
        time_str = f"{r['elapsed_s']/60:.0f}m"

        lines.append(
            f"{r['name']:<30} {status:>5} {time_str:>8} {best_wr:>7} {last_wr:>7} {n_evals:>6}"
        )

    lines.append("")
    lines.append("=" * 90)
    lines.append("")

    # Detailed EVAL history for each experiment
    lines.append("DETAILED EVAL HISTORY")
    lines.append("=" * 90)
    for r in results:
        lines.append(f"\n--- {r['name']} ({r['dir']}) ---")
        evals = collect_eval_lines(r["log"])
        if evals:
            for ev in evals:
                lines.append(f"  {ev}")
        else:
            lines.append("  (no EVAL lines found)")

    summary_text = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(summary_text)
    print(f"\nSummary written to: {summary_path}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <experiments.toml>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    common = cfg["common"]
    experiments = cfg["experiments"]

    es_train_exe, repo_root = find_es_train()
    common_args = build_common_args(common)
    output_dir = resolve_output_dir(common)

    print("Overnight ES Training Runner")
    print(f"Config:     {config_path}")
    print(f"Binary:     {es_train_exe}")
    print(f"Output dir: {output_dir}")
    print(f"Experiments: {len(experiments)}")
    time_limit = common.get("time_limit", "?")
    print(f"Time limit per experiment: {time_limit}s")
    print()

    # Verify executable exists
    if not os.path.isfile(es_train_exe):
        print(f"ERROR: es_train not found at {es_train_exe}")
        print("Build with: cargo build --release -p duke_training --bin es_train")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    results = []

    for i, exp in enumerate(experiments, 1):
        result = run_experiment(
            exp, i, len(experiments),
            es_train_exe, repo_root, common_args, output_dir,
        )
        results.append(result)

    total_elapsed = time.time() - total_start
    print(f"\n\nAll experiments complete in {total_elapsed:.0f}s ({total_elapsed/3600:.1f} hours)")

    write_summary(results, output_dir)


if __name__ == "__main__":
    main()
