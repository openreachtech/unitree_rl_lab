#!/usr/bin/env bash
# Loop 27 (v26) -- Stage B from Loop 26's model_24600, the best policy of the project.
#
# Loop 26 worked: rewarding the foot clearance -- the height a person actually sees, and
# the one height quantity the actuators do not cap -- moved it for the first time since
# Loop 18.
#     foot clearance  0.497 -> **0.600 m**   (+21%, and 0.607 at model_24700)
#     trunk rise      0.283 -> **0.301 m**
#     rear_first      0.012 -> **0.004**
#     distance        0.632 -> 0.646 m
#     v_z             2.731 -> 2.714          (unchanged, as it must be -- torque ceiling)
#     falls           0.05% -> 0.39%          (the cost)
#   model_24600 dominates model_22100, the previous clearance record, on every axis
#   (clearance 0.600 vs 0.603 but trunk 0.301 vs 0.282, rear_first 0.004 vs 0.059, falls
#   0.39% vs 1.14%).
#
# It collapsed after only 1250 iterations, earlier than the 1800-1900 of the previous nine,
#   and that turned out to be informative: **entropy carries across resumes.** It is the
#   policy's own action std, not optimiser state. Loop 26 started at entropy 6.7 -- already
#   decayed by Loop 24's entropy_coef of 0.002 -- and 0.002 drove it to 3.6, hitting the
#   too-deterministic failure that much sooner. The coefficient has to be read against
#   where the policy is entering, not chosen in the abstract.
#
# Change: entropy_coef 0.002 -> 0.005, the value that HELD entropy at 11.8-12.0 for 1800
#   iterations in Loop 25. A policy entering at 3.6 needs holding, not further narrowing.
#   No reward term is touched.
#
# Prediction
#   1. Entropy stops falling and settles somewhere above 5, instead of continuing toward 3.
#      This is the mechanism claim and it is visible within a few hundred iterations.
#   2. No collapse in 1200 iterations.
#   3. Foot clearance passes 0.618 m, the project peak, which Loop 26 missed by 0.5%.
#   4. Falls come back toward 0.05-0.2%. Loop 26 bought clearance partly out of the
#      landing; with the clearance term already at 0.92 of its 0.65 m target there is less
#      left for it to buy, so the landing terms should recover ground.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_11-07-40
LOAD_CKPT=model_24600.pt
ITERS=1200   # resume semantics: total = 24600 + 1200 = 25800
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 27 (v26) start: entropy_coef 0.002 -> 0.005 (entropy carries across resumes), from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 27 (v26) exited with code ${EXIT} ==="
exit "${EXIT}"
