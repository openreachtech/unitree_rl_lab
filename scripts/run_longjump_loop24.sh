#!/usr/bin/env bash
# Loop 24 (v23) -- Stage B from Loop 23's model_23100. The last untried mechanism.
#
# Loop 23 disproved the learning-rate hypothesis: the LR was pinned at 1e-4 for the whole
#   run and the policy still collapsed at iteration 23300, this time completely (every
#   metric to zero, falls 97%). It also produced the best landing of the height line on the
#   way there -- model_23100 at **0.77% falls**, clearance 0.531 m, trunk 0.275 m,
#   v_z 2.679, rear_first 0.040, success 96.5%.
#
# What all seven collapsed runs share is the signature, not the intervention:
#     entropy RISES (21.0 -> 22.4), estimator loss rises with it, then the policy goes.
#   Loops 21, 22 and 23 changed the landing band, the height targets and the learning rate
#   respectively, and all three broke identically 700-1200 iterations after resume. Rising
#   entropy is the opposite of convergence and is exactly what an entropy bonus does once
#   it is the largest live gradient left.
#
# It became the largest live gradient by design. Since Loop 22 both height terms are
#   targeted AT the machine ceiling -- the take-off torque has been pinned at the 45.43 N.m
#   knee limit for six loops -- so they are deliberately flat there. Flat task reward plus
#   entropy_coef 0.01 leaves the only thing still pushing being the term that widens the
#   action distribution, and a policy balanced on a saturated actuator has nowhere to widen
#   into.
#
# Change: entropy_coef 0.01 -> 0.002. Nothing else. Smoke on model_23100 shows entropy
#   already falling (20.67 -> 20.61) rather than climbing.
#
# Prediction
#   1. Entropy declines or holds for the whole run instead of rising past ~21.5. This is
#      the mechanism claim and it is visible long before any collapse would be.
#   2. No collapse in 2500 iterations, where seven consecutive runs broke by 1200.
#   3. Falls at or below 0.77%, since the landing terms are now the only live gradient and
#      they finally get an uninterrupted run at it.
#   4. If entropy still climbs and it collapses anyway, stop. Three reward-side and two
#      optimiser-side interventions will have failed identically, and the answer is that
#      this operating point cannot be trained further -- consolidate on model_23100.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_08-38-02
LOAD_CKPT=model_23100.pt
ITERS=2500   # resume semantics: total = 23100 + 2500 = 25600
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 24 (v23) start: entropy_coef 0.01 -> 0.002, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 24 (v23) exited with code ${EXIT} ==="
exit "${EXIT}"
