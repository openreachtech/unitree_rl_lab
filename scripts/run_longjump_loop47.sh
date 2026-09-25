#!/usr/bin/env bash
# Loop 47 -- the objective goes back to distance. From Loop 46's model_27800.
#
# tanaka: "飛距離のみを追い求めて" / "高さは気にしなくてよくて". So Loop 12's swap
# (distance -> apex) is undone: jump_takeoff_apex 50 -> 0, jump_takeoff_distance 0 -> 74.
#
# Loop 12 left distance because paying for the product 2*v_x*v_z/g buys the cheaper
# factor first, and at a 2.1 m/s approach that was v_x. Fourteen loops of the height
# objective have inverted the split -- v_x 1.665 against v_z 2.623 -- so the cheap factor
# now IS v_x, which is the one distance wants bought. The handover is income-neutral
# (31.9 -> 32.0) and restores the gradient: apex sat at 0.877 of full, distance at 0.432.
#
# One thing had to be fixed before re-enabling it: jump_takeoff_distance had no
# payout_window_s, so it paid for every step the jump window stayed open, which makes NOT
# LANDING profitable. Loop 16 regressed exactly that way on the apex term and Loop 17
# fixed it with a 0.6 s bound; the same bound is now on this term.
#
# Starting point is model_27800, which is Loop 46's best by mujoco (0.970 m distance,
# 20% landing over 40 trials) -- NOT model_27900, which led on the Isaac metric and came
# third in mujoco. Judge on mujoco.
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 47 start: objective -> distance (apex 50->0, distance 0->74), from model_27800, +1500 iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations 1500 --resume --load_run 2026-09-13_12-58-48 --checkpoint model_27800.pt \
  --deploy-keyboard-commands
echo "=== [$(ts)] Loop 47 exited with code $? ==="
