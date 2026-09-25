#!/usr/bin/env bash
# Control run for the Loop 32 speed sweep: same fixed approach, different starting policy.
#
# The sweep (model_27600, Loop 31's speed-conditioned policy) reads:
#     approach 1.67-1.77 m/s -> trunk rise 0.195, torque 44.8-45.5 N.m
#     approach 2.13-2.34 m/s -> trunk rise 0.206, torque 45.2-45.6 N.m
# Moving the approach by 0.6 m/s does not move the height, and both points sit on the
# 45.43 N.m knee limit. Meanwhile Loop 30's model_27000 reached trunk rise 0.286 at
# approach 2.26 m/s on LESS torque (44.19 N.m).
#
# Same speed, same (or more) torque, 0.08 m less height. That is not a speed/height trade
# -- the difference is in the take-off posture, and model_27600's is committed to putting
# the knee's impulse into horizontal velocity.
#
# This run tests exactly that, by changing ONE thing from the sweep's 2.6 point: the
# starting checkpoint. Everything else is identical -- 500 iterations, approach pinned at
# 2.6 m/s, approach gate disabled, same env count.
#
#   If trunk rise holds near 0.286 -> the starting posture decides the height, the speed
#   sweep was measuring the wrong variable, and the next loop must start from a
#   height-shaped policy and add speed to it rather than the reverse.
#
#   If it collapses to ~0.20 as well -> the height really is bought back by the slower
#   U(2.8, 3.3) command distribution rather than by the posture, and pinning the approach
#   is itself what costs the height.
#
# Either answer is worth having; the first one redirects the whole line of work.
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate

CFG=source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/longjump_env_cfg.py
BAK="${CFG}.bak_origin_control"
cp "${CFG}" "${BAK}"
restore() { cp "${BAK}" "${CFG}"; echo "=== cfg restored ==="; }
trap restore EXIT

python3 - "${CFG}" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
s = s.replace('"approach_lo": 2.0,', '"approach_lo": 0.0,').replace('"approach_hi": 2.5,', '"approach_hi": 0.1,')
s = re.sub(r"APPROACH_SPEED_MS: float = [\d.]+", "APPROACH_SPEED_MS: float = 2.6", s)
s = re.sub(r"APPROACH_SPEED_MIN_MS: float = [\d.]+", "APPROACH_SPEED_MIN_MS: float = 2.6", s)
open(p, "w", encoding="utf-8").write(s)
print("gate disabled, approach pinned at 2.6 m/s")
PY

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] control: model_27000 (Loop 30, height-shaped) at fixed approach 2.6 ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations 500 --resume --load_run 2026-09-09_17-14-30 --checkpoint model_27000.pt \
  --deploy-keyboard-commands
echo "=== [$(ts)] control done (exit $?) ==="
