#!/bin/bash
# Sweep all non-trivial features: train NN on each individual feature label
# to find which LR-Cheap heuristics are hardest for the NN to learn.

set -e
export PATH="/c/Users/Gal/.cargo/bin:$PATH"

INPUT="D:/temp/labeled_all_features.bin"
BASE_DIR="D:/temp/feature_sweep"
BINARY="J:/dev/git/duke_rust/target/release/supervised_train.exe"

# Non-trivial features (non-zero variance)
FEATURES=(0 1 2 3 4 5 6 7 8 10 12 13 14)
NAMES=(
    "near_my_duke_friendly"
    "near_my_duke_enemy"
    "near_enemy_duke_friendly"
    "near_enemy_duke_enemy"
    "my_moves"
    "opp_moves"
    "my_reachable"
    "opp_reachable"
    "contested"
    "my_threatened"
    "opp_threatened"
    "my_duke_mob"
    "opp_duke_mob"
)

mkdir -p "$BASE_DIR"

echo "=== Feature Sweep: training NN on each individual feature ==="
echo "Features to test: ${#FEATURES[@]}"
echo ""

for i in "${!FEATURES[@]}"; do
    fi=${FEATURES[$i]}
    name=${NAMES[$i]}
    dir="$BASE_DIR/feature_${fi}_${name}"
    log="$BASE_DIR/feature_${fi}_${name}.log"

    echo "--- Feature $fi: $name ---"
    mkdir -p "$dir"

    "$BINARY" \
        --input "$INPUT" \
        --hidden 128,64 \
        --lr 0.001 \
        --epochs 5 \
        --batch-size 256 \
        --eval-interval 50000000 \
        --eval-games 0 \
        --benchmark "base" \
        --checkpoint-dir "$dir" \
        --label-index "$fi" \
        --seed 42 \
        2>&1 | tee "$log"

    # Extract final epoch loss
    final_loss=$(grep "Epoch.*complete" "$log" | tail -1 | sed 's/.*avg_loss=//' | sed 's/,.*//')
    echo "  => Final loss: $final_loss"
    echo ""
done

# Summary
echo ""
echo "=== SUMMARY ==="
echo "Feature | Name | Final Loss"
echo "--------|------|----------"
for i in "${!FEATURES[@]}"; do
    fi=${FEATURES[$i]}
    name=${NAMES[$i]}
    log="$BASE_DIR/feature_${fi}_${name}.log"
    final_loss=$(grep "Epoch.*complete" "$log" | tail -1 | sed 's/.*avg_loss=//' | sed 's/,.*//')
    printf "%7d | %-30s | %s\n" "$fi" "$name" "$final_loss"
done
