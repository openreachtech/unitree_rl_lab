#!/usr/bin/env bash
# Loop 11 (v10) -- Stage B, resumed from Loop 10's best checkpoint (model_9400).
#
# ONE change: mdp.jump_takeoff_speed (weight 2.5 -> 0.0) is replaced by
# mdp.jump_takeoff_distance (weight 9.0), which rewards the ballistic distance the
# take-off has bought, 2*v_x*v_z/g, dense in the jump window and gated on real_jump.
#
# Why:
#   - The retired term was a constant by the end of Loop 10. It is clamp(v_x / 2.0, 0, 1)
#     and the policy took off at v_x = 2.158 m/s, i.e. pinned at 1.0 with zero gradient.
#     It did its job across Loops 8-10 (v_x 0.84 -> 2.16) and had nothing left to give.
#   - Loop 10 inverted the split it inherited: v_x 1.222 -> 2.158, v_z 1.624 -> 1.301.
#     Distance is the product, maximised at v_x = v_z for a given take-off energy, so at
#     C = 7.27 the ballistic distance is 0.572 m against 0.741 m for an even split --
#     30% sitting in the split. Loop 9 measured the mirror image (+4%) and correctly
#     left it alone; the lever that was empty then has re-opened.
#   - Rewarding the product makes the balance the physics' problem: dR/dv_x is
#     proportional to v_z and dR/dv_z to v_x, which is the objective's own sensitivity
#     ratio at every operating point. Raising jump_height's weight was the alternative
#     and was rejected -- apex height can also be bought by braking into the take-off,
#     the behaviour Loop 10 just removed, whereas the product cannot.
#   - Weight 9.0 hands over with the in-window income unchanged: 9.0 * (0.572 / 2.06)
#     = 2.50, the same flat 2.5 the retired term was paying.
#
# Prediction (one change, so one prediction):
#   v_z climbs back off 1.301 and the ballistic distance moves toward 0.74 m. v_x may
#   give back some of its 2.158 -- that is the trade being asked for, and only the
#   product matters. If v_z does NOT move, the constraint is mechanical (ground contact
#   time at a 2.1 m/s approach) rather than motivational, and the next lever is take-off
#   technique / EFGCL rather than any reward weight.
#
# Everything else is identical to Loop 10: assist off for the whole run, approach band
# U(2.8, 3.3), repeat_liftoff counted from the take-off.

set -uo pipefail

cd /home/tanaka/isaacsim/unitree_rl_lab

# Shared machine: /tmp/isaaclab/logs belongs to another user.
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"

# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate

LOAD_RUN=2026-09-05_02-31-01
LOAD_CKPT=model_9400.pt
# rsl_rl's --resume treats max_iterations as "total = resume_iter + this": 9400 -> 11400.
ITERS=2000

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "=== [$(ts)] Loop 11 (v10) start: takeoff_speed -> takeoff_distance (2*v_x*v_z/g, w 9.0), from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="

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

echo "=== [$(ts)] Loop 11 (v10) exited with code ${EXIT} ==="
exit "${EXIT}"
