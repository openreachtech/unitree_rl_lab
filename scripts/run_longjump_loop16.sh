#!/usr/bin/env bash
# Loop 16 (v15) -- Stage B from Loop 15's model_16000. The height term is about to die of
# its own success; this loop moves the target so it keeps working.
#
# What Loop 15 did (all four predictions hit)
#   Changing the landing detection from ">= 2 feet" to ">= 1 foot" -- one line -- turned
#   out to be worth more than the three reward loops before it:
#     unaided_timeout_fraction  0.86 -> 0.000       success_rate  ~0.15 -> 0.970
#     foot clearance            0.309 -> 0.432 m    (+40%)
#     v_z                       1.87  -> 2.301 m/s
#     rear_first_fraction       1.000 -> 0.503      front/rear foot gap 0.103 -> 0.000 m
#     bad_orientation           ~0.015 -> 0.004
#   So the "landing collapse" of Loops 12-14 really was a detection artifact, and the
#   landing-gear term from Loop 14 finished the job once the window stopped timing out.
#   The run was still improving at its final iteration for the first time in the project.
#
# Change 1 -- jump_takeoff_apex: target 0.28 -> 0.45 m, weight 9.3 -> 15.0.
#   At v_z = 2.301 the ballistic apex is 0.2694 m, i.e. 0.962 of the 0.28 m target. The
#   clamp is close enough that there is almost no gradient left -- the same death that
#   took jump_takeoff_speed in Loop 11, and this project's rule is to suspect saturation
#   before touching a weight. 0.45 puts it back at 0.599 with room above. The weight keeps
#   the handover income-neutral (9.3*0.962 = 8.95 out, 15.0*0.599 = 8.99 in) and the term
#   stays bounded at 1.0.
#
# Change 2 -- deploy_hold_time_s 0.65 -> 0.76, re-derived from unaided_window_length,
#   which Loop 15 added so this number stops being a stale estimate. The window settled at
#   0.762-0.767 s once landings were detected again. This only matters at the next mujoco
#   test, but it is the number that made Loop 7 look like a training failure.
#
# Prediction
#   1. v_z passes 2.301 and the foot clearance passes 0.436 m. If v_z stalls near 2.3
#      anyway, the constraint is no longer the reward and the next lever is mechanical --
#      take-off stance duration, or the approach speed (v_x has drifted 1.35 -> 1.06).
#   2. rear_first_fraction keeps falling below 0.503. The front/rear gap is already at
#      0.000 m, so the order should now be decided by the timing of the swing rather than
#      by the geometry, and the landing-gear term still pulls.
#   3. success_rate stays above 0.95 and timeout_fraction stays at 0. If either moves, the
#      higher jump has outgrown the landing again and that is a real wall this time.
#   4. Watch unaided_liftoff_vel_x (1.06). It has fallen every loop since Loop 12 as
#      height was bought; below 0.8 this stops being a running jump.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_02-35-48
LOAD_CKPT=model_16000.pt
ITERS=2500   # resume semantics: total = 16000 + 2500 = 18500
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 16 (v15) start: apex target 0.28 -> 0.45 (weight 15.0), deploy hold 0.76, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 16 (v15) exited with code ${EXIT} ==="
exit "${EXIT}"
