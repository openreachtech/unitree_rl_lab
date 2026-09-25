#!/usr/bin/env bash
# Loop 28 (v27) -- Stage B from Loop 27's model_25400.
#
# Loop 27 is the first run since Loop 17 that did not collapse, and it confirmed the
# entropy-carries-across-resumes reading:
#     entropy      4.10 -> 13.75, settled at ~13.5 and flat for the last 400 iterations
#     no collapse in 1200 iterations, where the previous ten broke at 700-1900
#     foot clearance  0.600 -> **0.627 m** (model_25200), a project record
#     best balance    model_25400: clearance 0.604, trunk 0.295, distance **0.687 m**,
#                     v_z 2.705, falls 0.34%, success 96.6%
#
# Change: jump_foot_clearance target 0.65 -> 0.80 m, weight 6.0 -> 7.4 (income-neutral:
#   6.0*0.965 = 5.79 out, 7.4*0.784 = 5.80 in). At 0.627 m the term is at 0.965 of its
#   target -- the saturation that killed jump_takeoff_speed in Loop 11 and jump_height in
#   Loop 18. Unlike jump_takeoff_apex this quantity is NOT capped by the actuators (it is
#   how the legs fold, not how hard the knees push), so its target should keep moving
#   rather than be pinned to a ceiling.
#
# entropy_coef stays at 0.005, which held entropy at 13.5 without collapse.
#
# Prediction
#   1. Foot clearance passes 0.627 m. This is the only height number still open.
#   2. No collapse in 1500 iterations, with entropy holding above 8.
#   3. Falls stay below 0.5% and trunk rise near 0.295 m -- clearance should keep coming
#      from the fold, not from trading the landing or the trunk for it.
#   4. Distance holds near 0.687 m. It has recovered from 0.632 as v_x came back to 1.36,
#      and it is the one number still far from its 2.06 m goal.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_11-36-41
LOAD_CKPT=model_25400.pt
ITERS=1500   # resume semantics: total = 25400 + 1500 = 26900
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 28 (v27) start: foot clearance target 0.65 -> 0.80 m (w 7.4), from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 28 (v27) exited with code ${EXIT} ==="
exit "${EXIT}"
