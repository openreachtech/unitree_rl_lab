#!/usr/bin/env bash
# Anaguma 試行6 (2026-09-14_20-00-36) の候補を MuJoCo で測る。
#
# この run で初めて Anaguma が補助なしで跳んだ (unaided_real_jump 0.73、高さ0.195m、滞空0.280s)。
# Isaac の数字は実機を予測しないので、候補はここで判定する。
#
#   A  model_2400   unaided real_jump 0.760 / 高さ 0.2052 / 飛距離 0.2623（高さ最良）
#   B  model_2800   unaided real_jump 0.766 / 高さ 0.1972 / 飛距離 0.2487（跳躍率最良）
#
# 助走はこのフェーズ1ポリシーの実測動作点 1.1 m/s で測る（学習時の unaided_trigger_speed 1.069）。
# A は 2.8 m/s でも測り、速い助走に載るかを見る。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
EXP=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1
RUN=${RUN_OVERRIDE:-2026-09-14_17-02-52}
OUT=${REPO}/${OUT_OVERRIDE:-eval_go2_loop}
SCENE=/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/go2/scene_flat.xml
MJPY=/home/tanaka/isaacsim/anaguma/venv/bin/python
cd "${REPO}"
export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

export_one() {
  local tag=$1 ckpt=$2
  local dir="${EXP}/${RUN}"
  local onnx="${OUT}/${tag}_${RUN}_${ckpt}.onnx"
  echo "=== [$(ts)] ${tag}: exporting ${ckpt}.pt ==="
  rm -f "${dir}/exported/policy.onnx"
  /home/tanaka/isaacsim/env_isaaclab/bin/python scripts/rsl_rl/play.py \
    --task Unitree-Go2-LongJump-v1 --headless --num_envs 1 \
    --checkpoint "${dir}/${ckpt}.pt" > "${OUT}/export_${tag}.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 90); do sleep 5; [ -s "${dir}/exported/policy.onnx" ] && break; done
  sleep 5; kill -9 "${pid}" 2>/dev/null; sleep 3
  [ -s "${dir}/exported/policy.onnx" ] || { echo "!!! export failed: ${OUT}/export_${tag}.log"; return 1; }
  cp "${dir}/exported/policy.onnx" "${onnx}"
  # 重み照合は IsaacLab の venv で回す（mujoco 用 venv に onnx が無い）。
  /home/tanaka/isaacsim/env_isaaclab/bin/python - "${onnx}" "${dir}/${ckpt}.pt" <<'PY'
import sys, numpy as np, onnx, torch
from onnx import numpy_helper
o, pt = sys.argv[1], sys.argv[2]
g = onnx.load(o).graph
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
sd = torch.load(pt, map_location="cpu")["model_state_dict"]
for key in ("actor.0.weight", "estimator.0.weight"):
    ref = sd[key].numpy()
    hit = [n for n, a in inits.items() if a.shape == ref.shape and np.allclose(a, ref, atol=1e-6)]
    print(f"    {key} {ref.shape} -> {'MATCH' if hit else '*** NO MATCH ***'}")
print("    input dim:", [d.dim_value for d in g.input[0].type.tensor_type.shape.dim])
PY
}

measure() {
  local tag=$1 ckpt=$2 vx=$3
  echo "=== [$(ts)] ${tag}: mujoco 40 trials (vx ${vx}) ==="
  "${MJPY}" scripts/mujoco_jump_eval.py --run "${EXP}/${RUN}" \
    --policy "${OUT}/${tag}_${RUN}_${ckpt}.onnx" --scene "${SCENE}" \
    --trials 40 --vx "${vx}" 2>&1 | tee "${OUT}/eval_${tag}_vx${vx}.log"
}

# 測る候補は環境変数で渡す: CANDIDATES="tag:ckpt:vx tag:ckpt:vx ..."
for spec in ${CANDIDATES:-"A:model_2400:1.1 A:model_2400:2.8 B:model_2800:1.1"}; do
  tag=${spec%%:*}; rest=${spec#*:}; ckpt=${rest%%:*}; vx=${rest##*:}
  [ -s "${OUT}/${tag}_${RUN}_${ckpt}.onnx" ] || export_one "${tag}" "${ckpt}" || continue
  measure "${tag}" "${ckpt}" "${vx}"
done
echo "=== [$(ts)] done ==="
