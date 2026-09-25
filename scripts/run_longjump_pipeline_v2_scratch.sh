#!/usr/bin/env bash
# Stage A v2 (forward-only) FROM SCRATCH -> Net2Net transplant -> Stage B, fully automated.
#
# 2026-09-02: tanaka氏の指示により、2026-08-31 17:41開始のStage A v2(forward-only,
# iter750/6000でシャットダウン処理中にcarb.tasking.Mutexアサーションでクラッシュ、
# 以降41時間放置)を model_700.pt から再開するのではなく、最初から(iteration 0)
# 再実行する。--max_iterations は Stage A/B とも 3000。Stage A完走後、自動で
# Net2Net移植 -> Stage B実行まで一気通貫で行う(セッション境界を跨いでも止まらない
# よう、このスクリプト自体をsetsid+nohup+disownで起動すること。
# 途中でどちらかのステージが失敗したら後続は起動しない。
#
# 参照: フィードバック_バックグラウンド学習ジョブの生存性.md
#       フィードバック_resume時カリキュラムリセット.md
#       フィードバック_共有tmpディレクトリの権限問題.md
#       プロジェクト_StageA完了とStageB移行.md

set -uo pipefail

cd /home/tanaka/isaacsim/unitree_rl_lab

export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"

# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "=== [$(ts)] Stage A v2 (forward-only) FROM SCRATCH, max_iterations=3000 ==="

python scripts/rsl_rl/train.py \
  --task Unitree-Go2-LongJump-Base-v1 \
  --headless \
  --num_envs 4096 \
  --max_iterations 3000 \
  --deploy-keyboard-commands
STAGE_A_EXIT=$?

echo "=== [$(ts)] Stage A v2 exited with code ${STAGE_A_EXIT} ==="

if [ "${STAGE_A_EXIT}" -ne 0 ]; then
  echo "!!! [$(ts)] Stage A v2 failed (exit ${STAGE_A_EXIT}) -- aborting pipeline, Stage B will NOT start."
  exit 1
fi

STAGE_A_RUN_DIR=$(ls -td logs/rsl_rl/unitree_go2_longjump_base_v1/*/ 2>/dev/null | head -1)
if [ -z "${STAGE_A_RUN_DIR}" ]; then
  echo "!!! [$(ts)] Could not locate Stage A run directory -- aborting."
  exit 1
fi
STAGE_A_CKPT=$(ls -v "${STAGE_A_RUN_DIR}"model_*.pt 2>/dev/null | tail -1)
if [ -z "${STAGE_A_CKPT}" ]; then
  echo "!!! [$(ts)] No Stage A checkpoint found in ${STAGE_A_RUN_DIR} -- aborting."
  exit 1
fi

echo "=== [$(ts)] Stage A v2 run dir: ${STAGE_A_RUN_DIR} ; final checkpoint: ${STAGE_A_CKPT} ==="

NET2NET_RUN_NAME="$(date '+%Y-%m-%d_%H-%M-%S')_net2net_v2scratch"
NET2NET_OUT_DIR="logs/rsl_rl/unitree_go2_longjump_v1/${NET2NET_RUN_NAME}"

echo "=== [$(ts)] Net2Net transplant -> ${NET2NET_OUT_DIR} ==="

python scripts/rsl_rl/longjump_net2net_transplant.py \
  --headless \
  --stage_a_checkpoint "${STAGE_A_CKPT}" \
  --output_dir "${NET2NET_OUT_DIR}"
NET2NET_EXIT=$?

echo "=== [$(ts)] Net2Net transplant exited with code ${NET2NET_EXIT} ==="

if [ "${NET2NET_EXIT}" -ne 0 ]; then
  echo "!!! [$(ts)] Net2Net transplant failed (exit ${NET2NET_EXIT}) -- aborting, Stage B will NOT start."
  exit 1
fi

echo "=== [$(ts)] Stage B (Unitree-Go2-LongJump-v1) from ${NET2NET_RUN_NAME}, max_iterations=3000 ==="

python scripts/rsl_rl/train.py \
  --task Unitree-Go2-LongJump-v1 \
  --headless \
  --num_envs 4096 \
  --max_iterations 3000 \
  --resume \
  --load_run "${NET2NET_RUN_NAME}" \
  --checkpoint model_0.pt \
  --deploy-keyboard-commands
STAGE_B_EXIT=$?

echo "=== [$(ts)] Stage B exited with code ${STAGE_B_EXIT} ==="
echo "=== [$(ts)] Pipeline finished. ==="
exit "${STAGE_B_EXIT}"
