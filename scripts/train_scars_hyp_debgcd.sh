#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
    echo "Usage: $0 CARS_ROOT PRETRAINED_PATH OUTPUT_DIR CHECKPOINT_DIR" >&2
    exit 2
fi

to_absolute() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$PWD" "$1" ;;
    esac
}

cars_root=$(to_absolute "$1")
pretrained_path=$(to_absolute "$2")
output_dir=$(to_absolute "$3")
checkpoint_dir=$(to_absolute "$4")

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir/.."

python -u train_HypDebGCD.py \
    --dataset_name scars \
    --cars_root "$cars_root" \
    --pretrained_path "$pretrained_path" \
    --output_dir "$output_dir" \
    --checkpoint_dir "$checkpoint_dir" \
    --dino v1 \
    --epochs 200 \
    --batch_size 128 \
    --num_workers 8 \
    --grad_from_block 10 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform imagenet \
    --lr 0.1 \
    --eval_funcs v2 \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1.0 \
    --sdl_loss_weight 0.005 \
    --adl_loss_weight 2.0 \
    --pl_loss_weight 0.5 \
    --threshold 0.7 \
    --c 0.1 \
    --clip_r 1.2 \
    --hyper_start_epoch 0 \
    --hyper_end_epoch 200 \
    --hyper_max_weight 1.0 \
    --hyper_temp_scale 0.3
