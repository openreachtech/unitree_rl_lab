#!/usr/bin/env bash
# Loop 31 (v30) -- make the jump pay in proportion to the approach speed.
# From Loop 30's model_27000.
#
# Loop 30 failed its actual verdict. The flight roll fell exactly as designed (1.350 ->
# 0.957 in Isaac, 0.668 measured in mujoco) and every Isaac number improved -- foot
# clearance 0.536 -> 0.619 m, past the 0.61 m that Loop 28 called a physical limit, with
# no change to the clearance reward at all; distance +6.4%; lift-off v_x +7.1%. And the
# mujoco landing rate went 53% -> 29% (4 of 14).
#
# So the roll was a correlate, not a cause -- the second one in a row after the take-off
# asymmetry. Twice now, "the quantity that separates mujoco's landings from its failures"
# has turned out not to be "the quantity that makes landings better when fixed".
#
# tanaka, jumping model_27000: "with a run-up the first one did not land; standing jumps
# succeeded several times. It is a RUNNING long jump, so jumping while running is what
# matters. And the running speed has got quite slow."
#
# That points at the task, not at the policy, and the task is wrong in a way nothing has
# checked in nineteen loops: **no reward has ever required the jump to happen while
# running**. jump_command fires irrespective of speed, and every jump reward -- this set's
# largest is jump_takeoff_apex at +1.68 -- pays a standing jump exactly what it pays a
# running one. A standing jump is the easier of the two, so that is what got learned.
# Loop 12 dropped v_x from the objective for a good reason (a product buys the cheap
# factor) and nothing put the approach back as a condition. Same shape as the
# APPROACH_SPEED_MS misreading in Loop 10.
#
# Isaac agrees the speed is short, though not that it got worse: the approach at the
# trigger measures 2.256 m/s against a U(2.8, 3.3) command -- 70-80% of what is asked for,
# in every loop. Loop 30 was actually slightly FASTER than Loop 29 (2.199 -> 2.256).
#
# Change: jump_takeoff_apex is multiplied by
#     gate = clamp((trigger_speed - 2.0) / 0.5, 0, 1)
# A gate rather than a product of the two objectives or a separate speed reward: a product
# lets the cheap factor be bought again, and a separate speed reward leaves the "sprint,
# stop, then jump" loophole. Scaling the jump's own payout makes a slow jump worth less AS
# A JUMP, with no way to collect the approach on its own.
#
# The ramp sits ABOVE the measured 2.256, not below: the smoke reads gate = 0.711 at the
# operating point (mean over envs of the per-env gate, so higher than the gate of the mean
# -- the fast tail already clips at 1.0), leaving real headroom. A gate that started
# saturated would be a dead term the way jump_takeoff_speed (Loop 11), jump_takeoff_apex
# (Loop 16) and jump_height (Loop 18) each were.
#
# The income drop is deliberate and NOT compensated by the weight: apex 1.68 -> 1.15 in
# the smoke. Compensating would restore exactly what this change exists to remove. Jumping
# still pays well against not jumping, so the cheapest response is to run faster.
#
# entropy_coef 0.002 -> 0.005, the same rule as always: chosen against the entropy the
# policy enters with. Loop 30's 0.002 narrowed model_27000 (action noise std 0.72 -> 0.38)
# and it then broke from the too-deterministic side at iter 27950 -- the Loop 26 -> 27
# situation exactly, where the holding value is what a narrow policy needs.
#
# jump_roll_rate stays at -4.0. It is not the change under test here, and every Isaac
# metric improved under it; whether it costs anything in mujoco is confounded with this
# loop and will need its own comparison later.
#
# Prediction
#   1. unaided_approach_gate rises from 0.711 past 0.85, i.e. the trigger-time approach
#      goes from 2.26 to at least 2.4 m/s.
#   2. The height side gives ground -- that trade is the point, and tanaka has accepted it.
#      The floor is foot clearance 0.50 (from 0.619). Below that this went too far.
#   3. mujoco lands a jump taken WITH a run-up.  <-- the actual verdict, needs tanaka
#   4. No collapse; entropy holds rather than falling to the deterministic side.
#
# Abort if entropy passes 16, or if foot clearance drops under 0.50.
#
# NOTE: --resume restores actor/critic weights only; lin_vel_cmd_levels resets, so the
# first ~50 iterations run at a near-zero approach and are not comparable
# (フィードバック_resume時カリキュラムリセット.md). Read nothing before iter 27050.
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-09_17-14-30
LOAD_CKPT=model_27000.pt
ITERS=1500   # resume semantics: total = 27000 + 1500 = 28500
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 31 (v30) start: approach gate on jump_takeoff_apex (2.0->2.5 m/s), entropy_coef 0.005, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 31 (v30) exited with code ${EXIT} ==="
exit "${EXIT}"
