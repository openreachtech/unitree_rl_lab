#!/usr/bin/env bash
# Measure the 2026-09-14 "from Loop10" candidate in MuJoCo (40 trials, vx 2.8).
#
#   D  2026-09-14_17-02-52 / model_10899   Isaac: unaided distance 0.799 m, real_jump 71%,
#                                          height 0.159, liftoff (2.11, 1.79). Plateaued
#                                          after iter 10400.
#
# The control is A 2026-09-13_14-19-07 / model_29200, measured the same way earlier today
# (40 trials, vx 2.8): 1.929 m, landing 12%. Its numbers are reused instead of re-measured.
# Isaac does not predict the machine, so D is judged only by the numbers below.
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
  /home/tanaka/isaacsim/env_isaaclab/bin/python - "${onnx}" "${dir}/${ckpt}.pt" <<'PY'
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

run_one D 2026-09-14_17-02-52 model_10899
echo "=== [$(ts)] done ==="
