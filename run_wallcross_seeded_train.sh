#!/usr/bin/env bash
# Local helper: train a wallcross task seeded from another wallcross task's checkpoint.
# Usage: ./run_wallcross_seeded_train.sh <task> <previous_task> <load_run> <checkpoint> <max_iterations> [extra train.py args]
cd "$(dirname "$0")"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/source/unitree_rl_lab" TMPDIR="$HOME/.cache/isaaclab_tmp"
task=$1; prev=$2; run=$3; ckpt=$4; iters=$5; shift 5
python -u scripts/rsl_rl/train_wallcross.py --task "$task" --num_envs 4096 --headless \
    --deploy-keyboard-commands --max_iterations "$iters" \
    --resume --previous-task "$prev" --load_run "$run" --checkpoint "$ckpt" "$@"
