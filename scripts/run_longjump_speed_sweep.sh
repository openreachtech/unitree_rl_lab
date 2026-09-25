#!/usr/bin/env bash
# Loop 32 -- measure the height-vs-approach-speed curve instead of guessing it.
#
# tanaka's proposal: "fix the running speed and the distance at their best values, and fix
# a height target for each run -- would that not jump higher?"
#
# Half of that is right and half has already been tried:
#
#   Fixing the operating point: SOUND. Loop 5 fixed the approach and it helped; Loop 12
#   removed v_x from the objective and v_z moved +37% in a single loop. Narrowing what the
#   policy has to cover concentrates it, and this project has evidence for that twice.
#
#   Raising a height target run by run: ALREADY DISPROVED on the v_z axis. Loop 16 raised
#   the apex target 0.28 -> 0.45, Loop 18 raised jump_height 0.28 -> 0.38, and Loop 21
#   showed what an unreachable target actually does -- v_z fell 2.70 -> 2.05, because a
#   gradient toward a speed the hardware cannot make walks the policy OFF the ceiling to
#   look for another route. Loop 22 fixed it by putting the target AT the measured ceiling.
#   The ceiling is the knee: 45.43 N.m.
#
# And Loop 31 shows why the trade is not negotiable by reward design. Take-off torque:
#     Loop 30 (approach 2.26 m/s):  44.19 N.m,  v_z 2.696,  trunk rise 0.286
#     Loop 31 (approach 2.54 m/s):  45.32 N.m,  v_z 2.483,  trunk rise 0.213
# The knee is saturated in Loop 31. Speed and height are not competing for reward budget,
# they are competing for the SAME TORQUE, and there is none spare. No weighting can buy
# both; only the operating point decides how the fixed impulse is split.
#
# So "fix the parameters at their optimal values" cannot be done by picking a number --
# the optimum is not known. This measures the curve so it can be picked knowingly, which
# is the same discipline that killed the Loop 30 symmetry design before it was built.
#
# Method: same start (Loop 31's model_27600), same budget (500 iters), one fixed approach
# speed per point, everything else identical. The approach gate on jump_takeoff_apex is
# DISABLED for the sweep (approach_lo/hi -> 0.0/0.1, so gate == 1): leaving it in would
# pay the slow points nothing and manufacture the very trend being measured.
#
# Output: trunk rise, foot clearance, lift-off v_z, distance and take-off torque at each
# fixed approach. tanaka then chooses the operating point.
#
# NOTE also worth having on record: Loop 28 concluded foot clearance had a physical limit
# of about 0.61 m. Loop 30 read 0.619-0.621 without touching the clearance reward at all,
# just by charging the flight roll. That limit was situational, not physical, and the
# height route was closed one loop early on that axis.
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate

CFG=source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/longjump_env_cfg.py
BAK="${CFG}.bak_speed_sweep"
cp "${CFG}" "${BAK}"
restore() { cp "${BAK}" "${CFG}"; echo "=== cfg restored ==="; }
trap restore EXIT

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
LOAD_RUN=2026-09-09_18-09-09
LOAD_CKPT=model_27600.pt

# Disable the approach gate for the whole sweep.
python3 - "${CFG}" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
s = s.replace('"approach_lo": 2.0,', '"approach_lo": 0.0,').replace('"approach_hi": 2.5,', '"approach_hi": 0.1,')
open(p, "w", encoding="utf-8").write(s)
print("approach gate disabled for sweep")
PY

for SPEED in 2.0 2.6 3.2 3.8; do
  python3 - "${CFG}" "${SPEED}" <<'PY'
import sys, re
p, v = sys.argv[1], sys.argv[2]
s = open(p, encoding="utf-8").read()
s = re.sub(r"APPROACH_SPEED_MS: float = [\d.]+", f"APPROACH_SPEED_MS: float = {v}", s)
s = re.sub(r"APPROACH_SPEED_MIN_MS: float = [\d.]+", f"APPROACH_SPEED_MIN_MS: float = {v}", s)
open(p, "w", encoding="utf-8").write(s)
print(f"approach fixed at {v} m/s")
PY
  echo "=== [$(ts)] sweep point: approach ${SPEED} m/s, 500 iters from ${LOAD_CKPT} ==="
  python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
    --max_iterations 500 --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
    --deploy-keyboard-commands > "SWEEP_${SPEED}.log" 2>&1
  echo "=== [$(ts)] sweep point ${SPEED} done (exit $?) ==="
done
echo "=== [$(ts)] sweep complete ==="
