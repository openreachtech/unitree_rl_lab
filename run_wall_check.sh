#!/usr/bin/env bash
# Local helper: run check_wall_terrain.py with the worktree's code (see play_hind.sh).
cd "$(dirname "$0")"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/source/unitree_rl_lab" TMPDIR="$HOME/.cache/isaaclab_tmp"
python -u scripts/rsl_rl/check_wall_terrain.py --task Go2-Multitask-WallCross-TerrainCheck \
    --checkpoint logs/rsl_rl/go2_multitask_biped_hind/2026-09-26_05-45-37/model_1499.pt --headless "$@"
