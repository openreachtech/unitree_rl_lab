#!/usr/bin/env bash
# Loop 45 (v42) -- penalise knees/elbows touching down. From Loop 32's model_27200.
#
# tanaka: "it sometimes lands on its knees and elbows -- add a penalty for that."
# Measured first: a knee or elbow touches on 99.8% of jumps. The existing always-on
# undesired_contacts (weight -1) pays -0.040 for that, 3% of jump_takeoff_apex, which is
# why nineteen loops trained knee-landing as standard practice. New jump-window term at
# -25.0 (smoke: -0.255), per-step so a scuff is cheap and a collapse is expensive.
#
# Context from the 51-jump mujoco trace of model_27200: landing quality is set by attitude,
# not by absorption. Post-landing max tilt separates success from failure at 2.03 standard
# deviations -- the sharpest discriminator measured in this project -- while knee flexion
# stroke separates at 0.20, i.e. not at all. That killed the "land in a crouch" idea and
# is consistent with three landing rewards in a row having failed.
#
# Abort: knee contact rate must be under 0.70 by iteration 27900, or the run stops. This is
# enforced by a watchdog rather than by judgement, because the previous loop's identical
# rule was overridden by hand and cost 946 iterations, -11% trunk rise and a 50x rise in
# falls. Height defences also enforced: trunk rise >=0.24, clearance >=0.58, v_z >=2.55.
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 45 (v42) start: approach PINNED at 2.8, track back to 1.5, from model_27200, +1500 iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations 1500 --resume --load_run 2026-09-09_17-14-30 --checkpoint model_27200.pt \
  --deploy-keyboard-commands
echo "=== [$(ts)] Loop 45 (v42) exited with code $? ==="
