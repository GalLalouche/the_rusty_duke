#!/bin/bash
# Overnight experiment: learn my_moves feature with various architectures
# Normalized FLPS labels, label index 4 (my_moves)
set -e

FC="/tmp/overnight_fc.exe"
CNN="/tmp/overnight_cnn.exe"
INPUT="D:/temp/labeled_all_features_norm.bin"
BASE="D:/temp/overnight_my_moves_$(date +%Y-%m-%d)"
LABEL_INDEX=4

mkdir -p "$BASE"
echo "=== Overnight my_moves experiment ==="
echo "Output: $BASE"
echo "Started: $(date)"
echo ""

# ── FC #1: 1024->1024->256x4 residual 2 ──
echo "--- FC: 1024->1024->256x4 residual 2 ---"
DIR="$BASE/fc_1024_1024_256x4_res2"
mkdir -p "$DIR"
"$FC" --input "$INPUT" --hidden 1024,1024,256,256,256,256 --residual 2 \
  --lr 0.001 --epochs 10 --batch-size 512 --eval-interval 50000000 \
  --eval-games 0 --benchmark base --checkpoint-dir "$DIR" \
  --label-index $LABEL_INDEX --seed 42 \
  2>&1 | tee "$BASE/fc_1024_1024_256x4_res2.log"
echo "DONE: $(date)"
echo ""

# ── FC #2: 1024->1024->256x4 residual 2 + sparse-init ──
echo "--- FC: 1024->1024->256x4 residual 2 + sparse-init ---"
DIR="$BASE/fc_1024_1024_256x4_res2_sinit"
mkdir -p "$DIR"
"$FC" --input "$INPUT" --hidden 1024,1024,256,256,256,256 --residual 2 --sparse-init \
  --lr 0.001 --epochs 10 --batch-size 512 --eval-interval 50000000 \
  --eval-games 0 --benchmark base --checkpoint-dir "$DIR" \
  --label-index $LABEL_INDEX --seed 42 \
  2>&1 | tee "$BASE/fc_1024_1024_256x4_res2_sinit.log"
echo "DONE: $(date)"
echo ""

# ── FC #3: 256x8 residual 2 ──
echo "--- FC: 256x8 residual 2 ---"
DIR="$BASE/fc_256x8_res2"
mkdir -p "$DIR"
"$FC" --input "$INPUT" --hidden 256,256,256,256,256,256,256,256 --residual 2 \
  --lr 0.001 --epochs 10 --batch-size 512 --eval-interval 50000000 \
  --eval-games 0 --benchmark base --checkpoint-dir "$DIR" \
  --label-index $LABEL_INDEX --seed 42 \
  2>&1 | tee "$BASE/fc_256x8_res2.log"
echo "DONE: $(date)"
echo ""

# ── FC #4: 256x8 residual 2 + sparse-init ──
echo "--- FC: 256x8 residual 2 + sparse-init ---"
DIR="$BASE/fc_256x8_res2_sinit"
mkdir -p "$DIR"
"$FC" --input "$INPUT" --hidden 256,256,256,256,256,256,256,256 --residual 2 --sparse-init \
  --lr 0.001 --epochs 10 --batch-size 512 --eval-interval 50000000 \
  --eval-games 0 --benchmark base --checkpoint-dir "$DIR" \
  --label-index $LABEL_INDEX --seed 42 \
  2>&1 | tee "$BASE/fc_256x8_res2_sinit.log"
echo "DONE: $(date)"
echo ""

# ── CNN #5: Box 32,32 + FC 256,128 ──
echo "--- CNN: Box 32,32 fc 256,128 ---"
DIR="$BASE/cnn_box_fc256_128"
mkdir -p "$DIR"
"$CNN" --input "$INPUT" --conv-channels 32,32 --fc-sizes 256,128 --kernel box \
  --lr 0.001 --epochs 25 --batch-size 256 --eval-interval 50000000 \
  --eval-games 0 --benchmark base --checkpoint-dir "$DIR" \
  --label-index $LABEL_INDEX --seed 42 --max-positions 1000000 \
  2>&1 | tee "$BASE/cnn_box_fc256_128.log"
echo "DONE: $(date)"
echo ""

# ── CNN #6: Diamond 32,32 + FC 256,128 ──
echo "--- CNN: Diamond 32,32 fc 256,128 ---"
DIR="$BASE/cnn_diamond_fc256_128"
mkdir -p "$DIR"
"$CNN" --input "$INPUT" --conv-channels 32,32 --fc-sizes 256,128 --kernel diamond \
  --lr 0.001 --epochs 25 --batch-size 256 --eval-interval 50000000 \
  --eval-games 0 --benchmark base --checkpoint-dir "$DIR" \
  --label-index $LABEL_INDEX --seed 42 --max-positions 1000000 \
  2>&1 | tee "$BASE/cnn_diamond_fc256_128.log"
echo "DONE: $(date)"
echo ""

# ── CNN #7: Cross 32,32 + FC 256,128 ──
echo "--- CNN: Cross 32,32 fc 256,128 ---"
DIR="$BASE/cnn_cross_fc256_128"
mkdir -p "$DIR"
"$CNN" --input "$INPUT" --conv-channels 32,32 --fc-sizes 256,128 --kernel cross \
  --lr 0.001 --epochs 25 --batch-size 256 --eval-interval 50000000 \
  --eval-games 0 --benchmark base --checkpoint-dir "$DIR" \
  --label-index $LABEL_INDEX --seed 42 --max-positions 1000000 \
  2>&1 | tee "$BASE/cnn_cross_fc256_128.log"
echo "DONE: $(date)"
echo ""

# ── Summary ──
echo "=== SUMMARY ==="
echo ""
for log in "$BASE"/*.log; do
  name=$(basename "$log" .log)
  final=$(grep "Epoch.*complete" "$log" | tail -1 | sed 's/.*avg_loss=//' | sed 's/,.*//')
  echo "$name: $final"
done
echo ""
echo "Finished: $(date)"
