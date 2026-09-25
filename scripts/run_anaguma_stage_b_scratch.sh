#!/usr/bin/env bash
# Anaguma Stage B, third attempt: from iteration 0 (fresh Net2Net) WITH the EFGCL assist.
#
# Attempt 1 (2026-09-13_15-26-37) was a clean Stage B from model_0 but inherited Go2's
# initial_assist_scale=0.0. Result: real_jump_fraction 0.0000 for all 3000 iterations.
# Attempt 2 (2026-09-14_14-55-23) turned the assist on but RESUMED from attempt 1's
# model_2999, and that was the wrong call. Measured over its first 400 iterations:
#       assist_scale 0.966 -> 0.763 (-21%)   real_jump 0.390 -> 0.108 (-72%)
#       unaided_max_height_gain 0.028, flat, never moved
# The policy shed jumps three times faster than the teacher was withdrawn. It had spent
# 3000 iterations learning that jump_command predicts no reward and that vertical motion
# costs some, so the assist was acting on weights already trained to cancel it.
#
# Hence iteration 0: the jump_command / jump_time columns are zero-initialised by the
# transplant, the action std starts fresh, and the teacher is present from the first step.
# Assist settings (scale 1.0, apex 0.15 m, crouch 243 N, fade over 1500 iterations) are in
# anaguma/longjump_env_cfg.py with the two smoke measurements that sized them.
#
# The objective stays HEIGHT (jump_takeoff_apex 50 / target 0.40, distance 0), as asked --
# unaffected by the same-day switch of Go2's objective to distance-only, because
# RewardsCfgAnagumaLongJump overrides both terms explicitly.
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
cd "${REPO}"
export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}"
PY=/home/tanaka/isaacsim/env_isaaclab/bin/python
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

# Both runs share one GPU. Transplants are the memory spike, so wait for Go2's to finish.
echo "=== [$(ts)] waiting for Go2's transplant to finish (.go2_transplant_done) ==="
WAITED=0
until [ -f "${REPO}/.go2_transplant_done" ]; do
  sleep 30; WAITED=$((WAITED+1))
  if [ "${WAITED}" -gt 60 ]; then echo "!!! [$(ts)] Go2 transplant did not finish in 30 min -- going ahead anyway."; break; fi
done
sleep 20

STAGE_A_CKPT="${REPO}/logs/rsl_rl/anaguma_longjump_base_v1/2026-09-13_13-35-30/model_2999.pt"
NET2NET_RUN_NAME="$(date '+%Y-%m-%d_%H-%M-%S')_net2net_anaguma_scratch"
NET2NET_OUT_DIR="logs/rsl_rl/anaguma_longjump_v1/${NET2NET_RUN_NAME}"

echo "=== [$(ts)] Net2Net transplant from ${STAGE_A_CKPT} ==="
mkdir -p "${NET2NET_OUT_DIR}"
timeout 900 "${PY}" scripts/rsl_rl/longjump_net2net_transplant.py --headless \
  --stage_a_checkpoint "${STAGE_A_CKPT}" --output_dir "${NET2NET_OUT_DIR}" \
  --stage_b_cfg unitree_rl_lab.tasks.locomotion.robots.anaguma.longjump_env_cfg:RobotEnvCfgAnagumaLongJump
echo "=== [$(ts)] transplant returned $? (124 = hung in shutdown, tolerated) ==="
pkill -9 -f "longjump_net2net_transplan[t].py" 2>/dev/null
sleep 5
if [ ! -s "${NET2NET_OUT_DIR}/model_0.pt" ]; then
  echo "!!! [$(ts)] ${NET2NET_OUT_DIR}/model_0.pt missing -- aborting."
  exit 1
fi

echo "=== [$(ts)] Anaguma Stage B from scratch (assist on from iter 0): ${NET2NET_RUN_NAME}, 3000 iters, 2048 envs ==="
"${PY}" scripts/rsl_rl/train.py --task Anaguma-LongJump-v1 --headless --num_envs 2048 \
  --max_iterations 3000 --resume --load_run "${NET2NET_RUN_NAME}" \
  --checkpoint model_0.pt --deploy-keyboard-commands
echo "=== [$(ts)] Anaguma Stage B from scratch exited with code $? ==="
