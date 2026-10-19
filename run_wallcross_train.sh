#!/usr/bin/env bash
# Local helper: train Go2-Multitask-WallCross from the reproduced hind-stance expert, with the
# worktree's code (see play_hind.sh). Same flags as train_and_aggregate.py, but through
# train_wallcross.py (which makes the task visible to train.py) and with live output.
# Usage: ./run_wallcross_train.sh <max_iterations> [extra train.py args]
cd "$(dirname "$0")"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/source/unitree_rl_lab" TMPDIR="$HOME/.cache/isaaclab_tmp"
iters="${1:-3000}"; shift || true
python -u scripts/rsl_rl/train_wallcross.py --task Go2-Multitask-WallCross --num_envs 4096 --headless \
    --deploy-keyboard-commands --max_iterations "$iters" \
    --resume --previous-task Go2-Multitask-Biped-Hind --load_run 2026-09-26_05-45-37 --checkpoint model_1499.pt "$@"
