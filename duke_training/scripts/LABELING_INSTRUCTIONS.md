# Labeling 50k LR-Cheap d3 Games on a Remote Machine

## Overview

Score 2.58M game positions with LR-Cheap depth-3 negamax evaluation.
Produces a `.scores` file (one f32 per position) used as training labels
for the HalfDA NNUE network.

**Input:** `trajectories.dtrj` (165MB, 50k games, 2.58M states)
**Output:** `scores_d3.bin` (9.9MB, 2.58M × 4 bytes)
**Estimated time:** ~2-3 hours on 8-core machine, ~1 hour on 24-core

---

## Step 0: Prerequisites

1. **Rust toolchain**: `rustup` installed with stable toolchain
2. **Git clone**: `git clone <repo_url>` and checkout the `claude` branch
3. **Copy these files to the remote machine:**
   - The entire `duke_rust` repo (or at least `src/` and `duke_training/`)
   - `D:/temp/halfda_d3_50k/trajectories.dtrj` (165MB)
   - `D:/temp/combined_lr_41.json` (the LR-Cheap evaluator weights, ~1KB)

---

## Step 1: Build

```bash
cd /path/to/duke_rust
cargo build --release -p duke_training --bin score_trajectories
```

Expected output: binary at `target/release/score_trajectories` (or `.exe` on Windows).
Build should take 1-3 minutes. No GPU needed.

**Verify the binary exists:**
```bash
ls -la target/release/score_trajectories*
# Should show a file ~10-20MB
```

---

## Step 2: Smoke Test (DO THIS FIRST)

Run on a tiny subset to verify everything works before committing to the full 3-hour run.

### 2a: Create a small test trajectory

We don't have a `--max-games` flag on score_trajectories, so we'll run on the
full file but kill it early. First, verify it starts correctly:

```bash
target/release/score_trajectories \
  --trajectories /path/to/trajectories.dtrj \
  --evaluator /path/to/combined_lr_41.json \
  --depth 3 \
  --output /tmp/test_scores.bin \
  2>&1 | head -20
```

**Expected output (first ~30 seconds):**
```
Loading trajectories from /path/to/trajectories.dtrj ...
Loaded 50000 games, 2581144 states in X.Xs
Loading LR-Cheap evaluator from /path/to/combined_lr_41.json ...
Scoring 2581144 positions with depth 3 negamax ...
  Scored 50000/2581144 (1.9%) [XXX pos/sec]
```

**Verify these EXACT numbers:**
- Games: `50000`
- States: `2581144`
- The evaluator loaded without error
- It's actually scoring (pos/sec > 0)

**Kill it after 30-60 seconds** (Ctrl+C).

### 2b: Verify the output file format

```bash
# Check file size: should be 4 bytes × (number of scored positions)
ls -la /tmp/test_scores.bin

# Check the first few scores are reasonable floats (not NaN or garbage)
python3 -c "
import struct
with open('/tmp/test_scores.bin', 'rb') as f:
    data = f.read(40)  # first 10 floats
n = len(data) // 4
vals = struct.unpack(f'<{n}f', data)
print(f'First {n} scores: {vals}')
# Scores should be in range [-30, 30]
# Most non-terminal scores are in [-5, 5]
# Terminal scores are exactly -30.0 or 30.0
# NaN means terminal or epsilon-random (but we're scoring, so should be rare)
for i, v in enumerate(vals):
    if v != v:  # NaN check
        print(f'  Score {i}: NaN (terminal state)')
    elif abs(v) > 30.01:
        print(f'  ERROR: Score {i} = {v} is out of range!')
    else:
        print(f'  Score {i}: {v:.4f} (OK)')
"
```

**Expected:** Scores between -30 and +30. Most between -5 and +5. No garbage values.

### 2c: Verify position count matches

```bash
python3 -c "
import os
size = os.path.getsize('/tmp/test_scores.bin')
num_scores = size // 4
print(f'Scores in file: {num_scores}')
# After full run, this should be exactly 2581144
# After smoke test (killed early), it should be some smaller number
# Each score is 4 bytes (f32 little-endian)
assert size % 4 == 0, f'ERROR: file size {size} is not a multiple of 4!'
print('Format OK: file size is a multiple of 4 bytes')
"
```

### 2d: Verify scoring rate

From the smoke test output, check the `pos/sec` number:
- **8-core machine:** expect ~100-300 pos/sec
- **16-core machine:** expect ~200-500 pos/sec
- **24-core machine:** expect ~400-800 pos/sec

Use this to estimate total time: `2581144 / (pos_per_sec) / 3600` hours.

---

## Step 3: Full Run

Once the smoke test passes, run the full scoring:

```bash
target/release/score_trajectories \
  --trajectories /path/to/trajectories.dtrj \
  --evaluator /path/to/combined_lr_41.json \
  --depth 3 \
  --output /path/to/scores_d3.bin \
  2>&1 | tee scoring_log.txt
```

Use `tee` to save the log. Use `nohup` or `screen`/`tmux` if running over SSH:

```bash
nohup target/release/score_trajectories \
  --trajectories /path/to/trajectories.dtrj \
  --evaluator /path/to/combined_lr_41.json \
  --depth 3 \
  --output /path/to/scores_d3.bin \
  > scoring_log.txt 2>&1 &
echo $! > scoring_pid.txt
tail -f scoring_log.txt
```

**Monitor progress:** The binary prints progress every 50k positions.

---

## Step 4: Post-Run Verification

After the run completes, verify the output:

### 4a: Check completion message

```bash
tail -5 scoring_log.txt
```

Expected:
```
  Scored 2581144/2581144 (100.0%) [XXX pos/sec]
Scoring complete in XXXs
Saved 2581144 scores to /path/to/scores_d3.bin (9.9 MB)
```

### 4b: Verify exact file size

```bash
python3 -c "
import os
size = os.path.getsize('/path/to/scores_d3.bin')
expected = 2581144 * 4  # exactly 2581144 f32 values
print(f'File size: {size} bytes')
print(f'Expected:  {expected} bytes')
assert size == expected, f'ERROR: size mismatch! Got {size}, expected {expected}'
print('SIZE CHECK PASSED')
"
```

**This MUST show `SIZE CHECK PASSED`.** If not, the run was interrupted or corrupted.

### 4c: Verify score distribution

```bash
python3 -c "
import struct, math

with open('/path/to/scores_d3.bin', 'rb') as f:
    data = f.read()

n = len(data) // 4
scores = struct.unpack(f'<{n}f', data)
print(f'Total scores: {n}')

# Count NaN, terminals, normal
nan_count = sum(1 for s in scores if math.isnan(s))
terminal_pos = sum(1 for s in scores if not math.isnan(s) and abs(s) >= 29.9)
terminal_neg = sum(1 for s in scores if not math.isnan(s) and abs(s) >= 29.9 and s < 0)
normal = [s for s in scores if not math.isnan(s) and abs(s) < 29.9]

print(f'NaN (terminal/random): {nan_count} ({nan_count/n*100:.1f}%)')
print(f'Terminal +-30: {terminal_pos} ({terminal_pos/n*100:.1f}%)')
print(f'Normal scores: {len(normal)} ({len(normal)/n*100:.1f}%)')

if normal:
    normal.sort()
    mean = sum(normal) / len(normal)
    print(f'Normal score range: [{normal[0]:.2f}, {normal[-1]:.2f}]')
    print(f'Normal score mean: {mean:.4f}')
    print(f'Percentiles:')
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        idx = min(int(len(normal) * p / 100), len(normal) - 1)
        print(f'  {p:3d}%: {normal[idx]:.4f}')

# Sanity checks
assert n == 2581144, f'Wrong count: {n}'
assert nan_count < n * 0.15, f'Too many NaN: {nan_count} ({nan_count/n*100:.1f}%)'
assert len(normal) > n * 0.80, f'Too few normal scores: {len(normal)}'
assert -5 < mean < 5, f'Mean too extreme: {mean}'
print()
print('ALL SANITY CHECKS PASSED')
"
```

**Expected output:**
- ~2-5% NaN (terminal states at end of games)
- ~2-5% terminal ±30 scores
- ~90%+ normal scores in range [-10, 10]
- Mean close to 0 (slight positive bias is OK since games are self-play)
- `ALL SANITY CHECKS PASSED`

---

## Step 5: Copy Back

Copy `scores_d3.bin` back to the main machine:

```bash
scp /path/to/scores_d3.bin user@main-machine:D:/temp/halfda_d3_50k/scores_d3.bin
```

Or use USB/network share. The file is only 9.9MB.

---

## Troubleshooting

### "Failed to load trajectories"
- Wrong file path, or file is corrupted/truncated
- Verify: `ls -la trajectories.dtrj` should show 164.8MB (172,810,240 bytes)

### "Failed to load CombinedWeights"
- Wrong path to `combined_lr_41.json`
- Verify: `cat combined_lr_41.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d), 'weights')"` should show `41 weights`

### "Scored 0/2581144 ... [0 pos/sec]" or hangs
- May need more stack space. The binary sets 8MB stacks via rayon ThreadPoolBuilder.
- Try: `RUST_MIN_STACK=16777216 target/release/score_trajectories ...`

### Scores are all NaN
- The evaluator isn't being called. Check that `--depth 3` is specified (not `--depth 0`).

### Scores are all 0.0
- Game result detection may be wrong. Check the trajectory file is the correct one (50k LR-Cheap d3 games, not random games).

### Process killed / OOM
- Loading 50k games into memory requires ~500MB RAM. Scoring adds ~40MB for scores.
- Total memory: ~600MB. Should work on any machine with 2GB+ free RAM.
