#!/usr/bin/env bash
# Local helper: continue a Go2-Multitask-WallCross run from its own checkpoint (same task lineage,
# so no --previous-task). The terrain curriculum is not persisted and restarts from the initial rows.
# Usage: ./run_wallcross_resume.sh <load_run> <checkpoint> <max_iterations> [extra train.py args]
cd "$(dirname "$0")"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/source/unitree_rl_lab" TMPDIR="$HOME/.cache/isaaclab_tmp"
run=$1; ckpt=$2; iters=$3; shift 3
python -u scripts/rsl_rl/train_wallcross.py --task Go2-Multitask-WallCross --num_envs 4096 --headless \
    --deploy-keyboard-commands --max_iterations "$iters" \
    --resume --load_run "$run" --checkpoint "$ckpt" "$@"
