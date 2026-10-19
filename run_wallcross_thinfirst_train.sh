#!/usr/bin/env bash
# Local helper: train Go2-Multitask-WallCross-ThinFirst seeded from a Go2-Multitask-WallCross run.
# Usage: ./run_wallcross_thinfirst_train.sh <load_run> <checkpoint> <max_iterations> [extra train.py args]
cd "$(dirname "$0")"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/source/unitree_rl_lab" TMPDIR="$HOME/.cache/isaaclab_tmp"
run=$1; ckpt=$2; iters=$3; shift 3
python -u scripts/rsl_rl/train_wallcross.py --task Go2-Multitask-WallCross-ThinFirst --num_envs 4096 --headless \
    --deploy-keyboard-commands --max_iterations "$iters" \
    --resume --previous-task Go2-Multitask-WallCross --load_run "$run" --checkpoint "$ckpt" "$@"
