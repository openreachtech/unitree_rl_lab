#!/usr/bin/env bash
# Export three long-jump candidates to ONNX and measure each in MuJoCo (40 trials).
#
# Isaac ranked these; Isaac does not predict the machine (2026-09-10: the policy that set
# an Isaac record jumped 0.120 m in mujoco and fell). So the candidates are judged here.
#
#   A  2026-09-13_14-19-07 / model_29200   Loop 47 best. ALREADY measured at 1.929 m over
#                                          40 trials -- included as the control, so B and C
#                                          are read against a number measured the same way
#                                          in the same session rather than against a note.
#   B  2026-09-13_15-14-43 / model_29400   Loop 48 peak. Isaac distance 1.247, falls 4.9%.
#   C  2026-09-13_16-41-28 / model_29500   Loop 49 peak. Isaac distance 1.209, falls 7.3%.
#
# model_29400 exists in FIVE different runs and model_27200 in two with different weights
# (リファレンス_ポリシーのDrive共有手順.md), so every artefact below is named with its run
# timestamp and each ONNX is checked against the .pt it came from before it is measured.
#
# play.py exports and then enters an endless render loop, so it is started detached and
# killed once exported/policy.onnx appears -- and because play.py always writes to
# <run>/exported/policy.onnx, each export is copied out under its own name immediately.
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
EXP=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1
OUT=${REPO}/eval_mujoco_20260914
MJPY=/home/tanaka/isaacsim/anaguma/venv/bin/python
cd "${REPO}"
export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

run_one() {
  local tag=$1 run=$2 ckpt=$3
  local dir="${EXP}/${run}"
  local onnx="${OUT}/${tag}_${run}_${ckpt}.onnx"
  echo "=== [$(ts)] ${tag}: exporting ${run}/${ckpt}.pt ==="
  rm -f "${dir}/exported/policy.onnx"
  /home/tanaka/isaacsim/env_isaaclab/bin/python scripts/rsl_rl/play.py \
    --task Unitree-Go2-LongJump-v1 --headless --num_envs 1 \
    --checkpoint "${dir}/${ckpt}.pt" > "${OUT}/export_${tag}.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 90); do sleep 5; [ -s "${dir}/exported/policy.onnx" ] && break; done
  sleep 5; kill -9 "${pid}" 2>/dev/null; sleep 3
  if [ ! -s "${dir}/exported/policy.onnx" ]; then
    echo "!!! [$(ts)] ${tag}: export failed -- see ${OUT}/export_${tag}.log"; return 1
  fi
  cp "${dir}/exported/policy.onnx" "${onnx}"
  echo "=== [$(ts)] ${tag}: onnx $(stat -c%s "${onnx}") bytes ==="

  # Does this ONNX actually hold the weights of the checkpoint it is named after?
  "${MJPY}" - "${onnx}" "${dir}/${ckpt}.pt" <<'PY'
import sys, numpy as np, onnx, torch
from onnx import numpy_helper
o, pt = sys.argv[1], sys.argv[2]
inits = {i.name: numpy_helper.to_array(i) for i in onnx.load(o).graph.initializer}
sd = torch.load(pt, map_location="cpu")["model_state_dict"]
ref = sd["actor.0.weight"].numpy()
hit = [n for n, a in inits.items() if a.shape == ref.shape and np.allclose(a, ref, atol=1e-6)]
print(f"    weight check: actor.0.weight {ref.shape} -> {'MATCH ' + hit[0] if hit else '*** NO MATCH ***'}")
print(f"    onnx input dim: {[d.dim_value for d in onnx.load(o).graph.input[0].type.tensor_type.shape.dim]}")
PY

  echo "=== [$(ts)] ${tag}: mujoco 40 trials (vx 2.8) ==="
  "${MJPY}" scripts/mujoco_jump_eval.py --run "${dir}" --policy "${onnx}" \
    --trials 40 --vx 2.8 2>&1 | tee "${OUT}/eval_${tag}.log"
}

run_one A 2026-09-13_14-19-07 model_29200
run_one B 2026-09-13_15-14-43 model_29400
run_one C 2026-09-13_16-41-28 model_29500
echo "=== [$(ts)] all done ==="
