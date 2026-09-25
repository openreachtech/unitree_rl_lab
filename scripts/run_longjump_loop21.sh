#!/usr/bin/env bash
# Loop 21 (v20) -- Stage B from Loop 20's model_22100.
#
# Loop 20 improved for 1000 iterations and then collapsed hard.
#   to iter 22500:  gap -0.152 -> -0.063 m, rear_first 0.04-0.09, clearance 0.60 m,
#                   trunk 0.30 m, v_z 2.69, falls ~1.1-1.5%
#   from iter 22700: falls 7.5% -> 35%, v_x 1.48 -> 0.87, clearance 0.41 m, success 0.88
#
# The two-sided gear term fixed the direction of the drift (the gap came back from -0.15
# toward -0.06) but never settled inside the target band, and the run then destabilised.
#
# Why, most likely: the symmetric ``|delta| - deadband`` band left delta in (0, +0.05] m
#   FREE -- and positive delta means the front feet are ABOVE the rear ones, which is
#   exactly the rear-first geometry that mdp.jump_landing_order charges at -4.0. The two
#   landing terms were disagreeing about a 5 cm band, and the landing budget as a whole
#   (gear -3.0, order -4.0, flight_pitch -4.0, feet_first +4.0) is now large enough for
#   that disagreement to matter -- v_x collapsing to 0.87 m/s says the running itself was
#   being squeezed.
#
# Change: the free band becomes ASYMMETRIC, [-0.06, 0] m. Front-above-rear is charged from
#   zero (agreeing with the order term instead of fighting it); front-below-rear is free to
#   6 cm, which is what a long jump should do, and charged past that. Smoke: gear income
#   -0.0135 against -0.0092 for the symmetric form, order -0.0028.
#
# Shorter run (2000 iterations, not 2500): Loop 20's useful window was its first 1000, and
#   every loop since 17 has produced its best checkpoint in the first half.
#
# Prediction
#   1. The gap settles inside [-0.06, 0] m rather than stopping at -0.10.
#   2. Falls go below 1% and stay there -- if they collapse again around 1000 iterations in,
#      the instability is not this band and the landing budget as a whole is too large.
#   3. rear_first stays below 0.10 with the gear term no longer pushing against the order
#      term.
#   4. v_x holds near 1.48. It is the early-warning signal: it fell before the falls did.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_06-27-36
LOAD_CKPT=model_22100.pt
ITERS=2000   # resume semantics: total = 22100 + 2000 = 24100
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 21 (v20) start: asymmetric landing band [-0.06, 0], from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 21 (v20) exited with code ${EXIT} ==="
exit "${EXIT}"
