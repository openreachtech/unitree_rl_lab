#!/usr/bin/env bash
# Loop 46 -- distance again. Raise the ASK for run-up speed, not the ceiling.
#
# tanaka: "走り幅跳びの飛距離重視の方がもう少し飛距離を伸ばせそう"。
#
# distance = 2*v_x*v_z/g. v_z has sat at 2.68-2.75 for fourteen loops and Loop 32 showed
# the knee torque is near its limit there. v_x is the factor that has moved every time it
# was asked for: Loop 31 (approach 2.26 -> 2.56, distance 0.701 -> 0.773) and Loop 32
# (2.78, distance 0.814, also a clearance record). Three points, no sign of a knee.
#
# Two changes, both on the ASK rather than on the ceiling:
#   * jump_takeoff_apex's approach gate 2.0/2.5 -> 2.6/3.1 (pays 0.36 of full at the
#     measured 2.78, so it has headroom and is not a dead term)
#   * approach command band back to U(2.8, 3.3) from Loop 45's pin at 2.8
#
# Explicitly NOT Loop 44's lever (cutting track_lin_vel_xy's weight). That released the
# approach to 4.4 m/s, set Isaac records, and those individuals landed in mujoco only
# 24-36% of the time. The tracking term stays at 1.5 so the approach still follows its
# command; what changes is what the command asks for.
#
# Judge on mujoco, not on Isaac (Loops 33-45: Isaac height/falls/roll predicted none of
# the landing rate). Success = distance above model_27200's 0.814 m with the mujoco
# landing rate still near its 42%.
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 46 start: approach band U(2.8,3.3), gate 2.6/3.1, from model_27200, +1500 iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations 1500 --resume --load_run 2026-09-09_17-14-30 --checkpoint model_27200.pt \
  --deploy-keyboard-commands
echo "=== [$(ts)] Loop 46 exited with code $? ==="
