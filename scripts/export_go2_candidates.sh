#!/usr/bin/env bash
# Go2 の mujoco 評価候補を ONNX に書き出す（測定はしない）。
#
#   CANDIDATES="tag:run:ckpt tag:run:ckpt ..." OUT_DIR=eval_go2_cycle4 \
#     bash scripts/export_go2_candidates.sh
#
# eval_go2_loop_mujoco.sh の export_one をそのまま切り出したもの。書き出し後に
# ONNX の重みと .pt の actor/estimator を照合する（照合は IsaacLab の venv で回す。
# mujoco 用 venv には onnx が無く、以前ここが黙って落ちていた）。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
EXP=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1
OUT=${REPO}/${OUT_DIR:-eval_go2_cycle4}
cd "${REPO}"
export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

export_one() {
  local tag=$1 run=$2 ckpt=$3
  local dir="${EXP}/${run}"
  local onnx="${OUT}/${tag}_${run}_${ckpt}.onnx"
  if [ -s "${onnx}" ]; then echo "=== [$(ts)] ${tag}: already exported, skip ==="; return 0; fi
  echo "=== [$(ts)] ${tag}: exporting ${run}/${ckpt}.pt ==="
  rm -f "${dir}/exported/policy.onnx"
  /home/tanaka/isaacsim/env_isaaclab/bin/python scripts/rsl_rl/play.py \
    --task Unitree-Go2-LongJump-v1 --headless --num_envs 1 \
    --checkpoint "${dir}/${ckpt}.pt" > "${OUT}/export_${tag}.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 90); do sleep 5; [ -s "${dir}/exported/policy.onnx" ] && break; done
  sleep 5; kill -9 "${pid}" 2>/dev/null; sleep 3
  [ -s "${dir}/exported/policy.onnx" ] || { echo "!!! export failed: ${OUT}/export_${tag}.log"; return 1; }
  cp "${dir}/exported/policy.onnx" "${onnx}"
  echo "    -> ${onnx}"
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

for spec in ${CANDIDATES}; do
  tag=${spec%%:*}; rest=${spec#*:}; run=${rest%%:*}; ckpt=${rest##*:}
  export_one "${tag}" "${run}" "${ckpt}"
done
echo "=== [$(ts)] all done ==="
ls -la "${OUT}"/*.onnx
