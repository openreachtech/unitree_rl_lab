#!/usr/bin/env bash
# 助走スイープ: 記録個体 1 本で、コマンド vx だけを振って mujoco で測る（学習なし）。
#
# 目的は「コマンドを上げると実測助走がいくつ上がるのか」を初めて測ること。
# これまでの Go2 評価 93 本は全て vx=2.8 の 1 点で、用量反応が未測定だった。
# 記録個体は vx=2.8 と言われて実測 3.37 m/s で走っており、コマンドは実測助走を
# 決めていない疑いがある。折り返し点（踏切が崩れるのか着地が先に死ぬのか）も見る。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
RUN=2026-09-15_18-53-20            # 記録個体 model_12700 (mujoco 平均 1.977 m)
ONNX=${REPO}/eval_go2_cycle14/Hc_${RUN}_model_12700.onnx
SCENE=/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/go2/scene_flat.xml
MJPY=/home/tanaka/isaacsim/anaguma/venv/bin/python
OUT=${REPO}/eval_go2_approach_sweep
cd "${REPO}"
export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

for vx in 2.0 2.4 2.8 3.2 3.6 4.0; do
  echo "=== [$(ts)] vx=${vx} 40試行 ==="
  "${MJPY}" scripts/mujoco_jump_eval.py --run "${REPO}/logs/rsl_rl/unitree_go2_longjump_v1/${RUN}" \
    --policy "${ONNX}" --scene "${SCENE}" --trials 40 --vx "${vx}" \
    > "${OUT}/sweep_vx${vx}.log" 2>&1
  tail -8 "${OUT}/sweep_vx${vx}.log"
done
echo "=== [$(ts)] done ==="
