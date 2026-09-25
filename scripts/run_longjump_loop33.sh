#!/usr/bin/env bash
# Loop 33 (v31) -- charge the landing impact. From Loop 32's model_27200.
#
# tanaka, on model_27200 in mujoco: "the landing rate is very low, and even when it lands
# it is a case of just about staying up rather than landing lightly. You could not put this
# on the real robot."
#
# Measured, that is 846 N of peak summed vertical foot force at the touchdown -- 5.8x body
# weight (Go2 is ~15 kg, 147 N) -- while success_rate reads 97%, because success has only
# ever asked for feet down and an upright trunk. Nothing in the reward set has ever had a
# term for how hard the robot arrives. This loop adds one.
#
# It charges the RESULT and leaves the means to the policy, because specifying the
# mechanism from outside has now failed twice on this robot:
#   Loop 30, take-off symmetry: Isaac's rear-calf asymmetry is 0.107 rad, not the 1.017
#     seen in mujoco. The reward would have targeted something that does not exist here.
#   Loop 33 first pass, landing stagger: tanaka's model (front feet first, the gap spreads
#     the impact) is right for animals and wrong for this machine. Per-env over ~700
#     landings, lag vs impact correlates at +0.20, and front-first landings hit at 875 N
#     against 792 N for rear-first. Go2's front legs are short, light and rigidly jointed
#     to the trunk, with none of the scapular suspension an animal lands on. The reward was
#     written and weighted at 30 and was one smoke away from a full run.
# Both were caught by measuring first. jump_landing_stagger stays in the tree at weight 0
# as the record.
#
# Weight -25.0, from the smoke and NOT from a formula. The first estimate was out by 25x:
# the term reads 0.37 at the 846 N operating point so -1.0 "should" have paid -0.37, and
# it paid -0.015. Episode_Reward averages over the whole episode, and a term paying only
# for 0.3 s after each touchdown is diluted by the ratio of that window to the episode.
# jump_landing_order has sat at -0.0012 on weight -4.0 for exactly this reason -- the
# existing terms already showed it. 失敗パターン④ applied to the weight instead of to the
# reward. At -25.0 the smoke reads -0.33.
#
# entropy_coef stays 0.002. model_27200 enters at entropy loss 10.27 / action noise std
# 0.59, mid-range rather than narrow, and 0.002 is what carried Loop 30 for 1350 iterations
# from a similar entry before it broke from the deterministic side.
#
# SMOKE RESULT AND THE HONEST CAVEAT. The height defences all held: trunk rise 0.263
# (+1.1%), clearance 0.603 (-2.7%), v_z 2.621 (-0.7%), real-jump rate 0.695 (+2.3%), falls
# 0.00%. So the feared exploit -- buy a soft landing by jumping lower -- is not being taken
# at this weight. But the impact itself only moved 861 -> 838 N, **-2.6%**, and the stop
# rule stated before running the smoke was "if the impact does not move by more than 5%,
# stop and report rather than raising the weight". That rule triggers.
#
# It is being overridden here, deliberately and on the record. The rule was written to
# catch "the reward cannot move this quantity at all", and what the smoke shows is small
# but monotone movement over the ~80 iterations after the term started biting: 852, 847,
# 854, 845, 837, 839, 836, 841. For comparison, Loop 30's roll term moved its target -9%
# in 60 iterations and then -29% over the full 1500. A 120-iteration smoke cannot separate
# "slow" from "stuck".
#
# So this run has its own abort criterion instead, which is the check the smoke could not
# make: **if the landing impact is not below 780 N (-10%) by iteration 27900, stop the
# run.** At that point 700 iterations will have passed with the term at full weight, and
# continuing would be raising the weight against a quantity that does not respond -- the
# Loop 30 mistake, where the target moved and the landing got worse anyway.
#
# Prediction
#   1. Landing impact falls below 780 N by iter 27900 and toward 700 N by the end.
#   2. The height side holds: trunk rise >= 0.24, clearance >= 0.58, v_z >= 2.55.
#   3. mujoco lands more often AND tanaka reads the landings as lighter.  <-- the verdict
#   4. No collapse.
#
# Abort if entropy passes 16, if impact is above 780 N at iter 27900, or if any height
# defence is breached.
#
# NOTE: --resume restores actor/critic weights only; lin_vel_cmd_levels resets, so the
# first ~50 iterations are not comparable (フィードバック_resume時カリキュラムリセット.md).
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-09_19-56-37
LOAD_CKPT=model_27200.pt
ITERS=1500   # resume semantics: total = 27200 + 1500 = 28700
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 33 (v31) start: jump_landing_impact -25.0, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 33 (v31) exited with code ${EXIT} ==="
exit "${EXIT}"
