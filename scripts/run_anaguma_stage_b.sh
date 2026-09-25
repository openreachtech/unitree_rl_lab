#!/usr/bin/env bash
# Anaguma: wait for Stage A to finish, transplant (Net2Net), then run Stage B.
#
# Detached and self-contained so the handover happens without anyone watching. Stage A is
# 3000 iterations from scratch; when its last checkpoint appears this transplants the
# actor/critic into a freshly built Stage B policy (the extra jump_command / jump_time
# inputs are zero-initialised, which is the point of Net2Net) and starts Stage B.
#
# The transplant script names Go2's Stage B cfg, and that is fine: the two machines have
# the same observation width (47) and the same 12 actions, so the network it builds is
# the right shape for Anaguma. What differs between them is the robot and the reward set,
# and those come from the task id passed to train.py below.
#
# Known quirk inherited from the Go2 pipeline: the transplant can hang in
# simulation_app.close() with the checkpoint already written (47 min observed 2026-09-02).
# So it is given a generous timeout and judged by whether model_0.pt exists, not by exit
# code.
set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

STAGE_A_RUN_DIR="logs/rsl_rl/anaguma_longjump_base_v1/2026-09-13_13-35-30/"
STAGE_A_LAST="${STAGE_A_RUN_DIR}model_2999.pt"
STAGE_B_ITERS=3000

echo "=== [$(ts)] waiting for Anaguma Stage A to finish (${STAGE_A_LAST}) ==="
WAITED=0
until [ -s "${STAGE_A_LAST}" ]; do
  sleep 60
  WAITED=$((WAITED + 1))
  if [ $((WAITED % 30)) -eq 0 ]; then echo "  [$(ts)] still waiting (${WAITED} min)"; fi
  if [ "${WAITED}" -gt 480 ]; then echo "!!! [$(ts)] Stage A did not finish in 8 h -- aborting."; exit 1; fi
done
sleep 30   # let the file finish being written
STAGE_A_CKPT=$(ls -v "${STAGE_A_RUN_DIR}"model_*.pt | tail -1)
echo "=== [$(ts)] Stage A done. Transplanting from ${STAGE_A_CKPT} ==="

NET2NET_RUN_NAME="$(date '+%Y-%m-%d_%H-%M-%S')_net2net_anaguma"
NET2NET_OUT_DIR="logs/rsl_rl/anaguma_longjump_v1/${NET2NET_RUN_NAME}"
mkdir -p "${NET2NET_OUT_DIR}"
timeout 900 python scripts/rsl_rl/longjump_net2net_transplant.py --headless \
  --stage_a_checkpoint "${STAGE_A_CKPT}" --output_dir "${NET2NET_OUT_DIR}" \
  --stage_b_cfg unitree_rl_lab.tasks.locomotion.robots.anaguma.longjump_env_cfg:RobotEnvCfgAnagumaLongJump
echo "=== [$(ts)] transplant returned $? (124 = hung in shutdown, tolerated) ==="
pkill -9 -f "longjump_net2net_transplant.py" 2>/dev/null
sleep 5
if [ ! -s "${NET2NET_OUT_DIR}/model_0.pt" ]; then
  echo "!!! [$(ts)] ${NET2NET_OUT_DIR}/model_0.pt missing -- aborting."
  exit 1
fi

echo "=== [$(ts)] Anaguma Stage B start: ${NET2NET_RUN_NAME}, ${STAGE_B_ITERS} iters ==="
python scripts/rsl_rl/train.py --task Anaguma-LongJump-v1 --headless --num_envs 2048 \
  --max_iterations "${STAGE_B_ITERS}" --resume --load_run "${NET2NET_RUN_NAME}" \
  --checkpoint model_0.pt --deploy-keyboard-commands
echo "=== [$(ts)] Anaguma Stage B exited with code $? ==="
