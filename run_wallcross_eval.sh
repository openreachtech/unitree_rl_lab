#!/usr/bin/env bash
# Local helper: evaluate a wall crossing checkpoint with check_wallcross_env.py (see run_wall_check.sh).
# usage: run_wallcross_eval.sh <checkpoint> <out_dir> [extra args...]
cd "$(dirname "$0")"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/source/unitree_rl_lab" TMPDIR="$HOME/.cache/isaaclab_tmp"
ckpt=$1; out=$2; shift 2
python -u scripts/rsl_rl/check_wallcross_env.py --checkpoint "$ckpt" --out "$out" --headless "$@"
