#!/usr/bin/env bash
# Loop 48 -- raise the distance target so the objective stops clamping.
#
# Loop 47 delivered: mujoco distance 0.970 -> 1.929 m over 40 trials (v_x 1.25 -> 2.97).
# But the ballistic product it is paid on now reads 1.66 m against a 2.06 m target --
# progress 0.806, the same place jump_takeoff_apex stalled at (0.877, Loop 36). A clamped
# term does not pull, so the target goes to 2.75 m with the weight raised to keep the
# payout per metre identical (74/2.06 = 35.9, 99/2.75 = 36.0). Only the ceiling moves.
#
# 2.75 is inside the machine's range, not aspirational: the energy method puts Go2 at
# 3.89 m for a 5.3 m/s approach and these policies already run at 5.00 m/s in mujoco.
# Loop 21 is the counterexample -- a target set past the ceiling walked v_z from 2.70
# down to 2.05 chasing it.
#
# Starting from model_29200, Loop 47's best BY MUJOCO (1.929 m, 12% landing). Isaac
# ranked 29100 first; mujoco put it third on distance. Judge on mujoco.
#
# Watch: landing rate is down to 12% from the 20% of Loop 46, and real-jump rate to 85%
# from 100%. Distance is what was asked for, so this continues, but if landing reaches
# zero the policy stops being a long jump and starts being a fall.
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 48 start: distance target 2.06 -> 2.75 (weight 74 -> 99), from model_29200, +1500 iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations 1500 --resume --load_run 2026-09-13_14-19-07 --checkpoint model_29200.pt \
  --deploy-keyboard-commands
echo "=== [$(ts)] Loop 48 exited with code $? ==="
