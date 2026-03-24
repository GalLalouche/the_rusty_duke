#!/usr/bin/env python3
"""Monitor running ES experiments and write a live summary to D:/temp/live_status.txt

Auto-discovers ES Training logs in D:/temp/ by scanning .log files for
'ES Training' in the first 5 lines.
"""

import os
import re
import time
from datetime import datetime

LOG_DIR = "D:/temp"
OUTPUT = "D:/temp/live_status.txt"
REFRESH_SECONDS = 30


def discover_logs(directory):
    """Scan directory for .log files containing 'ES Training' in first 5 lines."""
    logs = []
    try:
        entries = os.listdir(directory)
    except OSError:
        return logs

    for fname in sorted(entries):
        if not fname.endswith(".log"):
            continue
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "r", errors="replace") as f:
                head = []
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    head.append(line)
            if any("ES Training" in l for l in head):
                name = derive_name(fname, head)
                logs.append((name, fpath))
        except OSError:
            continue
    return logs


def derive_name(fname, head_lines):
    """Build a human-readable experiment name from filename and header."""
    base = fname.replace(".log", "")
    # Try to find network arch from header
    for line in head_lines:
        m = re.search(r"network:\s*([\d>-]+)", line)
        if m:
            arch = m.group(1)
            return f"{base} ({arch})"
    return base


def parse_log(logpath):
    """Parse all relevant data from a log file.

    Returns a dict with keys:
        status, iter_current, iter_total,
        base_evals, opp_evals, iter_lines,
        sigma_adaptations, opp_eps_changes
    """
    result = {
        "status": "not started",
        "iter_current": None,
        "iter_total": None,
        "base_evals": [],
        "opp_evals": [],
        "iter_lines": [],
        "sigma_adaptations": [],
        "opp_eps_changes": [],
    }

    if not os.path.exists(logpath):
        return result

    try:
        with open(logpath, "r", errors="replace") as f:
            content = f.read()
    except OSError:
        return result

    if not content.strip():
        result["status"] = "empty"
        return result

    # Status
    if "Time limit reached" in content or "training complete" in content.lower():
        result["status"] = "DONE"
    elif os.path.getmtime(logpath) < time.time() - 120:
        result["status"] = "STALE"
    else:
        result["status"] = "running"

    # EVAL lines (vs base heuristic)
    for m in re.finditer(
        r"EVAL:.*iter=(\d+).*A=([\d.]+)%.*B=([\d.]+)%.*Tie=([\d.]+)%", content
    ):
        result["base_evals"].append(
            (int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
        )

    # VS_OPP lines
    for m in re.finditer(
        r"VS_OPP:.*iter=(\d+).*A=([\d.]+)%.*B=([\d.]+)%.*Tie=([\d.]+)%", content
    ):
        result["opp_evals"].append(
            (int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
        )

    # iter lines: iter  650/100000: avg_wr+=0.497 avg_wr-=0.484 max_wr=0.850 sigma=0.1600 opp_eps=0.00 (...)
    for m in re.finditer(
        r"iter\s+(\d+)/(\d+):\s+avg_wr\+=([\d.]+)\s+avg_wr-=([\d.]+)\s+max_wr=([\d.]+)\s+sigma=([\d.]+)\s+opp_eps=([\d.]+)",
        content,
    ):
        result["iter_lines"].append({
            "iter": int(m.group(1)),
            "total": int(m.group(2)),
            "avg_wr_plus": float(m.group(3)),
            "avg_wr_minus": float(m.group(4)),
            "max_wr": float(m.group(5)),
            "sigma": float(m.group(6)),
            "opp_eps": float(m.group(7)),
        })

    # Current iteration from last iter line
    if result["iter_lines"]:
        last = result["iter_lines"][-1]
        result["iter_current"] = last["iter"]
        result["iter_total"] = last["total"]

    # Sigma adapted lines: "Sigma adapted: 0.0400 (no improvement for 3 evals)"
    for m in re.finditer(r"Sigma adapted:\s*([\d.]+)\s*\(([^)]+)\)", content):
        result["sigma_adaptations"].append((float(m.group(1)), m.group(2).strip()))

    # Opponent epsilon lines: "Opponent epsilon: 0.40 -> 0.35 (training wr was 69.5%)"
    for m in re.finditer(
        r"Opponent epsilon:\s*([\d.]+)\s*->\s*([\d.]+)\s*\(([^)]+)\)", content
    ):
        result["opp_eps_changes"].append(
            (float(m.group(1)), float(m.group(2)), m.group(3).strip())
        )

    return result


def format_output(logs_data):
    """Format all parsed data into the output string."""
    lines = []
    lines.append(f"=== Live Status \u2014 {datetime.now().strftime('%H:%M:%S')} ===")
    lines.append("")

    # Summary table header
    lines.append(
        f"{'Experiment':<40} {'Status':<10} {'Iter':>6} {'Sigma':>7} {'OppEps':>7} "
        f"{'TrainWR':>8} {'BestBase':>9} {'LastBase':>9}"
    )
    lines.append("-" * 100)

    for name, logpath, data in logs_data:
        status = data["status"]
        iter_str = ""
        sigma_str = ""
        opp_eps_str = ""
        train_wr_str = ""
        best_base_str = ""
        last_base_str = ""

        if data["iter_current"] is not None:
            iter_str = str(data["iter_current"])

        if data["iter_lines"]:
            last_iter = data["iter_lines"][-1]
            sigma_str = f"{last_iter['sigma']:.4f}"
            opp_eps_str = f"{last_iter['opp_eps']:.2f}"
            # Train WR: use avg_wr+ of last iteration as representative
            train_wr_str = f"{last_iter['avg_wr_plus'] * 100:.1f}%"

        if data["base_evals"]:
            best_base = max(e[1] for e in data["base_evals"])
            last_base = data["base_evals"][-1][1]
            best_base_str = f"{best_base:.1f}%"
            last_base_str = f"{last_base:.1f}%"

        lines.append(
            f"{name:<40} {status:<10} {iter_str:>6} {sigma_str:>7} {opp_eps_str:>7} "
            f"{train_wr_str:>8} {best_base_str:>9} {last_base_str:>9}"
        )

    lines.append("")

    # Detailed sections for each experiment
    for name, logpath, data in logs_data:
        has_detail = (
            data["base_evals"]
            or data["opp_evals"]
            or data["iter_lines"]
            or data["sigma_adaptations"]
            or data["opp_eps_changes"]
        )
        if not has_detail:
            continue

        # Base evals
        if data["base_evals"]:
            lines.append(f"--- {name} vs Base ---")
            for iter_n, win, loss, tie in data["base_evals"]:
                lines.append(
                    f"  iter {iter_n:>4}: Win={win:>5.1f}% Loss={loss:>5.1f}% Tie={tie:>5.1f}%"
                )
            lines.append("")

        # Opponent evals
        if data["opp_evals"]:
            lines.append(f"--- {name} vs Training Opponent ---")
            for iter_n, win, loss, tie in data["opp_evals"]:
                lines.append(
                    f"  iter {iter_n:>4}: Win={win:>5.1f}% Loss={loss:>5.1f}% Tie={tie:>5.1f}%"
                )
            lines.append("")

        # Training progress (last 10 iter lines)
        if data["iter_lines"]:
            lines.append(f"--- {name} Training Progress (last 10) ---")
            recent = data["iter_lines"][-10:]
            for il in recent:
                lines.append(
                    f"  iter {il['iter']:>5}: avg_wr+={il['avg_wr_plus']:.3f} "
                    f"avg_wr-={il['avg_wr_minus']:.3f} max_wr={il['max_wr']:.3f} "
                    f"sigma={il['sigma']:.4f} opp_eps={il['opp_eps']:.2f}"
                )
            lines.append("")

        # Sigma adaptations
        if data["sigma_adaptations"]:
            lines.append(f"--- {name} Sigma Adaptations ---")
            for new_sigma, reason in data["sigma_adaptations"]:
                lines.append(f"  sigma -> {new_sigma:.4f} ({reason})")
            lines.append("")

        # Opponent epsilon changes
        if data["opp_eps_changes"]:
            lines.append(f"--- {name} Opponent Epsilon Changes ---")
            for old_eps, new_eps, reason in data["opp_eps_changes"]:
                lines.append(f"  eps {old_eps:.2f} -> {new_eps:.2f} ({reason})")
            lines.append("")

    return "\n".join(lines)


def main():
    while True:
        try:
            # Discover logs dynamically each cycle
            logs = discover_logs(LOG_DIR)

            # Parse all logs
            logs_data = []
            for name, logpath in logs:
                data = parse_log(logpath)
                logs_data.append((name, logpath, data))

            # Format and write
            output = format_output(logs_data)
            try:
                with open(OUTPUT, "w") as f:
                    f.write(output)
            except OSError:
                pass

        except Exception as e:
            # Never crash - write error to status file
            try:
                with open(OUTPUT, "w") as f:
                    f.write(
                        f"=== Live Status \u2014 {datetime.now().strftime('%H:%M:%S')} ===\n\n"
                        f"ERROR: {e}\n"
                    )
            except OSError:
                pass

        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
