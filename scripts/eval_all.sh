#!/usr/bin/env bash
# Evaluate all best checkpoints on 4 ImageNet variants.
#
# Selects the highest-scoring best-*.ckpt from each experiment directory
# and runs zero-shot eval on imagenet / imagenet_v2 / imagenet_r / imagenet_sketch.
#
# Usage:
#   bash scripts/eval_all.sh
#   bash scripts/eval_all.sh 2>&1 | tee eval_results.log
#
# Prerequisites:
#   - imagenet-sketch-wds-full already created:
#       python scripts/imagenet_imagefolder_to_wds.py \
#           --src  $DATA/sketch \
#           --out  $DATA/imagenet-sketch-wds-full
#   - Virtual environment activated (or adjust PYTHON below)

set -euo pipefail

RESULTS=/net/storage/pr3/plgrid/plggwie/plgmazurekagh/results_clip_kd
DATA=/net/storage/pr3/plgrid/plggwie/plgmazurekagh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

# ImageNet variant paths
IN1K=$DATA/imagenet-v1-wds-full
INV2=$DATA/imagenet-v2-wds/test
INR=$DATA/imagenet-r-wds/test
INSK=$DATA/imagenet-sketch-wds-full

# Experiment directories to evaluate
EXPERIMENTS=(
    teacher_pretrained_vit_b_16
    student_baseline_vit_t_16
)

echo "=========================================="
echo "Zero-shot ImageNet evaluation"
echo "Results dir : $RESULTS"
echo "Data dir    : $DATA"
echo "=========================================="

for exp in "${EXPERIMENTS[@]}"; do
    ckpt_dir="$RESULTS/$exp/checkpoints"

    last="$ckpt_dir/last.ckpt"
    if [[ ! -f "$last" ]]; then
        echo "WARNING: last.ckpt not found in $ckpt_dir, skipping"
        continue
    fi

    echo ""
    echo "--- Experiment : $exp"
    echo "    Checkpoint : $last"
    echo ""

    $PYTHON "$SCRIPT_DIR/eval_open_clip.py" \
        lightning_ckpt="$last" \
        imagenet_val="$IN1K" \
        imagenet_v2="$INV2" \
        imagenet_r="$INR" \
        imagenet_sketch="$INSK"
done

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
