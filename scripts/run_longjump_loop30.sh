#!/usr/bin/env bash
# Loop 30 (v29) -- charge the flight roll rate. From Loop 29's model_26600.
#
# Loop 29 turned out to have worked: mujoco went from 0/8 landings to 23/43 (53%). The
# number Loop 29 named as "the one that has to move", unaided_peak_pitch_rate, never
# moved -- it is the push-off transient, and what decides the landing is the steady
# rotation after it. That is name-vs-measured #11, and the first instance in a diagnostic
# used to make decisions rather than in a reward or a setting.
#
# 43 traced jumps say two components of the handed-over angular momentum separate the
# landings from the failures about equally: the signed flight pitch rate (81% of attempts
# classified by its sign alone) and the flight roll rate (79%). Flight time and jump
# height separate nothing (0.03 and 0.12 standardised difference) -- nothing is decided
# in the air.
#
# The first design attacked the cause: mujoco's rear calves sit 1.017 rad apart at
# lift-off, correlating with the roll at r = -0.86. Measuring the same quantity in Isaac
# BEFORE writing that reward killed it -- at the matched operating point (approach
# 2.19 m/s, real_jump 0.744) Isaac reads 0.107 rad, nine times smaller, while the flight
# roll rate is 1.350 rad/s, HIGHER than mujoco's failure group. The roll is not produced
# by the asymmetry, and a symmetry reward would have spent a whole loop on a quantity
# that does not exist on this side.
#
# What is left is a quantity that IS present in Isaac, at a level mujoco punishes and
# Isaac does not (0.4% falls at 1.35 rad/s), and that nothing in the reward set charges.
#
# Change: jump_roll_rate (weight -4.0, max_rate 2.2), bounded, charged only while airborne
# inside a real_jump. Magnitude rather than signed -- the failures roll both ways, so it is
# two-sided by construction and cannot run away the way Loop 19's relu() penalties did.
#
# Weight chosen from the smoke, not from a formula. At -4.0 the term reads -0.092 at the
# operating point against jump_takeoff_apex's +1.861, i.e. 4.9% -- five times the income of
# jump_flight_pitch (-0.018) at the identical weight, which is the term that has never been
# large enough to shape a take-off. Decisive evidence for keeping -4.0 rather than raising
# it: the measured roll ALREADY moved inside the 60-iteration smoke, 1.368 -> 1.251. The
# gradient is live at this weight, and raising it would be a second unmeasured change.
#
# entropy_coef 0.005 -> 0.002 is the established rule, not a new experiment: the
# coefficient is chosen against the entropy the policy enters with. model_26600 enters at
# 12.6, and Loop 29 put the holding value (0.005) on exactly that policy and watched
# entropy climb to 20.2 and collapse after 700 iterations.
#
# Prediction
#   1. unaided_flight_roll_rate falls from 1.35 to below 1.0.
#   2. mujoco landing rate beats 53%.  <-- the actual verdict, needs tanaka at the sim
#   3. Height holds: trunk rise ~0.285, lift-off v_z ~2.67, clearance ~0.52. A loss here
#      means suppressing the roll is costing take-off energy, which is a real finding.
#   4. No collapse; entropy falls from 12.6 rather than climbing.
#
# Abort if entropy passes 16.
#
# NOTE: --resume restores actor/critic weights only. lin_vel_cmd_levels resets, so the
# first ~50 iterations run at a near-zero approach speed and are not comparable
# (フィードバック_resume時カリキュラムリセット.md). Read nothing before iter 26650.
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-07_00-04-25
LOAD_CKPT=model_26600.pt
ITERS=1500   # resume semantics: total = 26600 + 1500 = 28100
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 30 (v29) start: jump_roll_rate -4.0, entropy_coef 0.002, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 30 (v29) exited with code ${EXIT} ==="
exit "${EXIT}"
