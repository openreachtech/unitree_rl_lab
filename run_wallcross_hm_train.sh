#!/usr/bin/env bash
# Local helper: train a height-map wallcross task from a widened run in its own log root.
# Usage: ./run_wallcross_hm_train.sh <task> <load_run> <checkpoint> <max_iterations> [extra train.py args]
cd "$(dirname "$0")"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/source/unitree_rl_lab" TMPDIR="$HOME/.cache/isaaclab_tmp"
task=$1; run=$2; ckpt=$3; iters=$4; shift 4
python -u scripts/rsl_rl/train_wallcross.py --task "$task" --num_envs 4096 --headless \
    --deploy-keyboard-commands --max_iterations "$iters" \
    --resume --load_run "$run" --checkpoint "$ckpt" "$@"
