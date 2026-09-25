#!/usr/bin/env bash
# Anaguma Stage A -- flat-ground running from scratch.
#
# The port of the Go2 long-jump task onto ALTs' Anaguma. Stage A only: get the machine
# running on flat ground, which is the Net2Net base the jump stage is built on.
#
# What this run is actually testing: the internal doc「AnagumaとGo2の違い」states that
# applying the unitree_rl_lab settings unchanged leaves the robot "unable to walk at
# all". The two causes it names are handled -- base_link not being at the CoM (no
# base-height reward is in this task's set, so nothing reads the wrong datum) and the
# solver iteration counts (8/4, set in the robot config) -- so if it still cannot walk,
# the cause is something neither doc records.
#
# Watch track_lin_vel_xy and Curriculum/lin_vel_cmd_levels. Go2's Stage A reached 3.2 m/s
# at iteration 7100. Anaguma has legs 1.33x as long but a joint-speed ceiling of 45%, and
# the internal doc reports it reaching 5.66 m/s with Go2's controller, so the speed is
# expected to arrive -- slowly at first, because 24.3 kg is 1.6x the mass Go2's reward
# weights were tuned against.
#
# 2048 envs rather than Go2's 4096: Loop 46 is training on the same GPU.
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Anaguma Stage A start: 2048 envs, 3000 iters ==="
python scripts/rsl_rl/train.py --task Anaguma-LongJump-Base-v1 --headless --num_envs 2048 \
  --max_iterations 3000
echo "=== [$(ts)] Anaguma Stage A exited with code $? ==="
