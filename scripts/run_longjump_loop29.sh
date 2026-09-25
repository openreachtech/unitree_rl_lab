#!/usr/bin/env bash
# Loop 29 (v28) -- Stage B from Loop 28's model_26400.
#
# Loop 28 closed the height route: trunk rise 0.301 m, lift-off v_z 2.72 m/s and foot
# clearance 0.61 m all hit their own ceilings, and two consecutive runs finished without
# collapsing at entropy_coef 0.005. In Isaac the policy lands: falls 0.42%, upright at
# window close 0.888. In mujoco on stock settings it never landed once across eight
# traced jumps.
#
# The traces say why, and it is not the reward set:
#   - The tilt starts growing at t = 0.4-0.5 s after the trigger, once joint speeds have
#     dropped to 1-9 rad/s -- i.e. after lift-off, with the legs essentially still. The
#     rotation is angular momentum handed over by the take-off, and Go2 cannot shed it in
#     flight (the legs carry 79% of the pitch inertia, but swinging them is a one-shot
#     trade and folding vs extending moves I by only 18%).
#   - Isaac's own flight rotation for the SAME policy is 0.48 rad/s RMS (backed out of a
#     jump_pitch_rate probe at weight -1.0). mujoco sustains ~2 rad/s for half a second.
#   - mujoco's knees reach 37-58 rad/s in the push-off. go2.xml's ctrlrange is flat in
#     speed; Isaac's actuator derates from X1 = 13.5 rad/s to zero at X2 = 30. So the two
#     simulators disagree about how much torque exists exactly where the pitch angular
#     momentum is created.
#
# Porting that curve INTO mujoco was tried on 2026-09-06 and rejected: landings got worse
# (2 of 4 attempts ended inverted, flight |w_y| rose to 1.87-2.99 rad/s), and matching the
# evaluator to the trainer removes the only independent check there is. tanaka's call, and
# it is the right one: mujoco stays stock and harsh, and the training side is what widens.
#
# Change: randomise the actuator torque-speed curve per RESET.
#   torque_scale (Y1, Y2)  U(0.80, 1.20)
#   speed_scale  (X1, X2)  U(0.80, 2.50)   -> X2 spans 24 to 75 rad/s
# At the top of the speed range there is no effective derate over the operating range, so
# mujoco's flat ctrlrange sits INSIDE the training distribution instead of outside it.
# Per reset rather than per startup on purpose: the existing mass / gain / friction terms
# fix a machine for the whole run, which a policy with an observation history can still
# identify online. This is the property the take-off was found to depend on, so it has to
# stay unlearnable. Verified live in a smoke -- forcing torque_scale to 2.0 moved
# unaided_peak_torque_frac from 44.8 to 72.2 N.m.
#
# entropy_coef stays at 0.005 and no reward term changes, so anything that moves is this.
#
# Prediction
#   1. unaided_peak_pitch_rate falls below the 4.8-6.5 rad/s it has held for ten loops.
#      This is the number that has to move for mujoco to land at all.
#   2. upright_at_close rises above 0.888; falls stay under 0.5%.
#   3. Trunk rise gives up a little (0.28-0.30 m from 0.301) and lift-off v_z drops toward
#      2.6 m/s -- the policy can no longer tune the push-off to one exact torque ceiling.
#      A loss here is acceptable; the height route is already saturated.
#   4. No collapse in 1500 iterations, entropy holding above 8.
#
# NOTE: --resume restores only the actor/critic weights. lin_vel_cmd_levels and the other
# curricula reset to their initial values, so the first few hundred iterations look easier
# than they are -- do not read the early reward jump as convergence
# (フィードバック_resume時カリキュラムリセット.md).

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_12-00-05
LOAD_CKPT=model_26400.pt
ITERS=1500   # resume semantics: total = 26400 + 1500 = 27900
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 29 (v28) start: actuator torque-speed curve DR per reset, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 29 (v28) exited with code ${EXIT} ==="
exit "${EXIT}"
