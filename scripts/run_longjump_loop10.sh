#!/usr/bin/env bash
# Loop 10 (v9) -- Stage B, resumed from v8's best checkpoint (model_7600).
#
# Three changes, all from the 2026-09-04 post-mortem in docs/longjump_presentation.md:
#
#   1. mdp.jump_repeat_liftoff now counts surplus flights from the TAKE-OFF's flight
#      phase instead of from the first flight phase of the window (検討6). Measured on
#      model_7600, the take-off was the 2nd liftoff of the window in 5 of 5 sampled
#      iterations, so the term was charging -0.368/s in-window against jump_height's
#      +0.494/s -- i.e. it taxed every real jump, and taxed the long flights hardest.
#   2. EFGCL assist off for the entire run (検討5). Its launch is sized for a 0.12 m
#      apex (v0 = 1.53 m/s); model_7600 already takes off unaided at v_z = 1.62 m/s for
#      0.119 m, so the teacher has nothing left to demonstrate. Loop 9 had only 400 of
#      its 1500 iterations fully unaided, and the unaided numbers fell throughout them.
#   3. Instrumentation for where the approach energy goes (検討3): trigger speed ->
#      take-off stance entry speed -> lift-off speed. No reward reads these.
#   4. Approach band lin_vel_x U(0.0, 3.3) -> U(2.8, 3.3), which change 3 is what
#      justified. Measured at the trigger of real jumps: the robot was approaching at
#      1.11 m/s, not the 3.3 the whole task was written around (3.3 was the ceiling of a
#      uniform range) -- and it kept 100% of it through the take-off, so the "62% lost
#      braking" premise of Loop 8 and 9 was arithmetic on an approach that never
#      happened. distance = 2*v_x*v_z/g is linear in v_x; the run-up is the lever.
#      Verified in a 64-env smoke: trigger speed 1.11 -> 1.9 m/s, lift-off v_x
#      1.21 -> 1.81, take-off energy 5.0 -> 6.6, no terminations.
#
# Prediction, checkable against tensorboard:
#   - Rewards/jump_repeat_liftoff goes to ~0 (it should now fire only on real bounces).
#   - unaided_takeoff_energy rises from 4.13; distance beats 0.359 m.
#   - unaided_trigger_speed settles near the 2.8-3.3 band rather than at 1.1, and
#     stance_entry -> lift-off stays a small loss, confirming the take-off is not where
#     the energy goes.
#   - The risk to watch is the opposite trade: v_z fell 1.57 -> 1.40 in the smoke as v_x
#     rose. Distance is the product, so watch 2*v_x*v_z/g, not either factor.

set -uo pipefail

cd /home/tanaka/isaacsim/unitree_rl_lab

# Shared machine: /tmp/isaaclab/logs belongs to another user (see
# フィードバック_共有tmpディレクトリの権限問題.md).
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"

# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate

LOAD_RUN=2026-09-04_02-37-20
LOAD_CKPT=model_7600.pt
# rsl_rl's --resume treats max_iterations as "total = resume_iter + this", so this is
# the number of NEW iterations: 7600 -> 9600.
ITERS=2000

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "=== [$(ts)] Loop 10 (v9) start: repeat_liftoff counted from take-off, assist 0.0, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="

python scripts/rsl_rl/train.py \
  --task Unitree-Go2-LongJump-v1 \
  --headless \
  --num_envs 4096 \
  --max_iterations "${ITERS}" \
  --resume \
  --load_run "${LOAD_RUN}" \
  --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?

echo "=== [$(ts)] Loop 10 (v9) exited with code ${EXIT} ==="
exit "${EXIT}"
