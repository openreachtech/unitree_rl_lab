#!/usr/bin/env bash
# Loop 32, the missing cell: the HEIGHT-shaped policy at a HIGH fixed approach.
#
# What is known after the speed sweep and the origin control, all at 500 iterations,
# approach pinned, approach gate disabled, 4096 envs -- identical except where noted:
#
#                        approach ~2.0      approach ~3.3
#   fast posture         height 0.193       height 0.238     (model_27600, sweep)
#   height posture       height 0.256       ** NOT MEASURED **  (model_27000, control)
#
# Two separate effects are established and neither has been combined with the other:
#   - WITHIN a posture, more approach speed gives MORE height, not less. The sweep moved
#     the approach 1.64 -> 3.34 m/s and height rose 0.193 -> 0.238, distance 0.669 ->
#     0.876, v_z 2.39 -> 2.60, all monotonic. The "speed costs height" premise this loop
#     started from is wrong inside a posture.
#   - ACROSS postures, model_27000 holds ~0.25 where model_27600 holds ~0.21 under the
#     same command, and it does so on LESS torque (44.1-45.4 vs a flat 45.4 ceiling) while
#     tracking the velocity command at only 73-79% instead of 88-91%.
#
# If both effects hold together, this run reads about 0.29-0.30, against an all-time
# project best of 0.312 (model_21200, Loop 19). If the height instead decays toward 0.21,
# then a high speed command is what destroys the posture -- which is Loop 31's failure
# stated as a mechanism rather than as a guess, and it is the evidence needed before
# spending a loop on any posture-holding reward.
#
# This is deliberately a measurement and not a design. Two design bets have missed on this
# project (Loop 30's take-off symmetry, and the "the approach gate is what cost the height"
# inference that the sweep then contradicted), while both measurements paid -- Step 0
# stopped a reward aimed at an asymmetry that does not exist in Isaac, and the sweep
# overturned the speed/height trade-off outright. The 500 iterations here decide between
# three different next loops instead of committing to one of them blind.
#
# Watch the height TRAJECTORY, not just its mean: where inside the 500 iterations the
# posture gives way (if it does) is what says whether ramping the speed command slowly is
# a real option or merely a slower version of the same failure.
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate

CFG=source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/longjump_env_cfg.py
BAK="${CFG}.bak_high_posture_fast"
cp "${CFG}" "${BAK}"
restore() { cp "${BAK}" "${CFG}"; echo "=== cfg restored ==="; }
trap restore EXIT

python3 - "${CFG}" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
s = s.replace('"approach_lo": 2.0,', '"approach_lo": 0.0,').replace('"approach_hi": 2.5,', '"approach_hi": 0.1,')
s = re.sub(r"APPROACH_SPEED_MS: float = [\d.]+", "APPROACH_SPEED_MS: float = 3.8", s)
s = re.sub(r"APPROACH_SPEED_MIN_MS: float = [\d.]+", "APPROACH_SPEED_MIN_MS: float = 3.8", s)
open(p, "w", encoding="utf-8").write(s)
print("gate disabled, approach pinned at 3.8 m/s")
PY

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] high posture (model_27000) at fixed approach 3.8 -- the missing cell ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations 500 --resume --load_run 2026-09-09_17-14-30 --checkpoint model_27000.pt \
  --deploy-keyboard-commands
echo "=== [$(ts)] missing cell done (exit $?) ==="
