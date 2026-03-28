#!/usr/bin/env python3
"""Monitor running ES experiments and append live stats to D:/temp/live_status.txt

Auto-discovers ES Training logs in D:/temp/ by scanning .log files for
'ES Training' in the first 5 lines.

Only shows RUNNING experiments. Each cycle appends a compact timestamp block
with the latest stats. Truncates the file when it exceeds 10000 lines.
"""

import os
import re
import time
from datetime import datetime

LOG_DIR = "D:/temp"
OUTPUT = "D:/temp/live_status.txt"
REFRESH_SECONDS = 30
MAX_LINES = 10000
KEEP_LINES = 5000


def discover_logs(directory):
    """Scan directory and subdirectories for .log files containing 'ES Training' in first 5 lines."""
    logs = []
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            if not fname.endswith(".log"):
                continue
            fpath = os.path.join(root, fname)
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


def truncate_if_needed(filepath):
    """If the file exceeds MAX_LINES, truncate to the last KEEP_LINES lines."""
    try:
        with open(filepath, "r", errors="replace") as f:
            all_lines = f.readlines()
    except (OSError, FileNotFoundError):
        return

    if len(all_lines) > MAX_LINES:
        keep = all_lines[-KEEP_LINES:]
        with open(filepath, "w") as f:
            f.write("--- (truncated) ---\n")
            f.writelines(keep)


def format_compact(logs_data):
    """Format a compact append block for running experiments only."""
    # Filter to running only
    running = [(name, logpath, data) for name, logpath, data in logs_data
               if data["status"] == "running"]

    if not running:
        return None

    lines = []
    lines.append(f"--- {datetime.now().strftime('%H:%M:%S')} ---")

    for name, logpath, data in running:
        # Build the summary line
        iter_str = ""
        sigma_str = ""
        opp_eps_str = ""
        train_wr_str = ""
        bench_str = ""

        if data["iter_current"] is not None:
            iter_str = f"iter={data['iter_current']}"

        if data["iter_lines"]:
            last_iter = data["iter_lines"][-1]
            sigma_str = f"sigma={last_iter['sigma']:.4f}"
            opp_eps_str = f"opp_eps={last_iter['opp_eps']:.2f}"
            train_wr_str = f"train_wr={last_iter['avg_wr_plus'] * 100:.1f}%"

        if data["base_evals"]:
            last_base = data["base_evals"][-1][1]
            bench_str = f"bench={last_base:.1f}%"

        parts = [p for p in [iter_str, sigma_str, opp_eps_str, train_wr_str, bench_str] if p]
        lines.append(f"{name} {' '.join(parts)}")

        # Last eval line if available
        if data["base_evals"]:
            last_eval = data["base_evals"][-1]
            lines.append(
                f"  Last eval (iter {last_eval[0]}): "
                f"Win={last_eval[1]:.1f}% Loss={last_eval[2]:.1f}% Tie={last_eval[3]:.1f}%"
            )

    return "\n".join(lines) + "\n"


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

            # Truncate file if it's gotten too large
            truncate_if_needed(OUTPUT)

            # Format compact block for running experiments only
            block = format_compact(logs_data)

            if block:
                try:
                    with open(OUTPUT, "a") as f:
                        f.write(block)
                except OSError:
                    pass

        except Exception as e:
            # Never crash - append error to status file
            try:
                with open(OUTPUT, "a") as f:
                    f.write(
                        f"--- {datetime.now().strftime('%H:%M:%S')} ---\n"
                        f"ERROR: {e}\n"
                    )
            except OSError:
                pass

        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
