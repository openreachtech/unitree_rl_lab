#!/usr/bin/env bash
# Anaguma Stage B, second attempt -- with the EFGCL assist switched on.
#
# The 2026-09-13 run (2026-09-13_15-26-37) walked well and never jumped: real_jump_fraction
# 0.0000 for all 3000 iterations. It inherited Go2's initial_assist_scale=0.0, which is
# right for a Go2 that already jumps and wrong for a machine that has never left the
# ground, because every jump reward sits behind the real_jump gate. See the assist block
# in anaguma/longjump_env_cfg.py for the smoke measurements behind the 0.15 setting.
#
# Starts from that run's model_2999, so the running approach is kept and only the take-off
# has to be learned. rsl_rl's --resume makes tot_iter = 2999 + max_iterations, so 4000 here
# runs to 6999. force_zero_at_step=36_000 (1500 iterations) withdraws the assist over the
# first 1500, leaving 2500 fully unaided -- judgement is on the unaided_* metrics only.
#
# Detached with setsid so it survives the session that started it
# (フィードバック_バックグラウンド学習ジョブの生存性.md).
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "=== [$(ts)] Anaguma Stage B (assist on, apex 0.15, decay over 1500 iter) start: 2048 envs, +4000 iters from model_2999 of 2026-09-13_15-26-37 ==="
/home/tanaka/isaacsim/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Anaguma-LongJump-v1 --headless --num_envs 2048 \
  --max_iterations 4000 --resume --load_run 2026-09-13_15-26-37 \
  --checkpoint model_2999.pt --deploy-keyboard-commands
echo "=== [$(ts)] Anaguma Stage B (assist on) exited with code $? ==="
