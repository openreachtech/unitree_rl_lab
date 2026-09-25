#!/usr/bin/env bash
# mujoco(C++/DDS 経路)で動かすポリシーを差し替える。go2 / anaguma 共通。
#
#   bash scripts/mujoco_load_policy.sh anaguma 2026-09-15_03-50-07 model_10100
#   bash scripts/mujoco_load_policy.sh go2     2026-09-15_14-38-44 model_12200
#
# やること:
#   1) <run>/exported/policy.onnx が無ければ play.py で書き出す
#   2) その run の deploy.yaml に keyboard_vel_scale: 1.0 を入れる(既定0.5だと助走が半分)
#   3) 実験ディレクトリ直下の exported/params symlink を張り替える(ctrl は直下しか見ない)
#   4) ONNX 入力次元 と deploy.yaml の観測合計 を突き合わせる(不一致は例外を出さずに壊れる)
set -euo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
ROBOT=${1:?usage: mujoco_load_policy.sh <go2|anaguma> <run> <model_XXXX>}
RUN=${2:?run dir name}
CKPT=${3:?model_XXXX}
case "${ROBOT}" in
  anaguma) EXPROOT=${REPO}/logs/rsl_rl/anaguma_longjump_v1; TASK=Anaguma-LongJump-v1 ;;
  go2)     EXPROOT=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1; TASK=Unitree-Go2-LongJump-v1 ;;
  *) echo "robot は go2 か anaguma"; exit 1 ;;
esac
DIR=${EXPROOT}/${RUN}
[ -f "${DIR}/${CKPT}.pt" ] || { echo "ない: ${DIR}/${CKPT}.pt"; exit 1; }
cd "${REPO}"; export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}"

# --- 1) ONNX
want="${DIR}/exported/${CKPT}.onnx"
if [ ! -s "${want}" ]; then
  echo "=== ${CKPT} を書き出し中 (play.py) ==="
  rm -f "${DIR}/exported/policy.onnx"
  /home/tanaka/isaacsim/env_isaaclab/bin/python scripts/rsl_rl/play.py \
    --task "${TASK}" --headless --num_envs 1 --checkpoint "${DIR}/${CKPT}.pt" \
    > "${TMPDIR}/export_${RUN}_${CKPT}.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 90); do sleep 5; [ -s "${DIR}/exported/policy.onnx" ] && break; done
  sleep 5; kill -9 "${pid}" 2>/dev/null || true; sleep 2
  [ -s "${DIR}/exported/policy.onnx" ] || { echo "書き出し失敗: ${TMPDIR}/export_${RUN}_${CKPT}.log"; exit 1; }
  cp "${DIR}/exported/policy.onnx" "${want}"       # ckpt 名つきで保存(同名別物の事故防止)
fi
cp -f "${want}" "${DIR}/exported/policy.onnx"
rm -f "${DIR}/exported/policy.pt"
printf '%s/%s  loaded %s\n' "${RUN}" "${CKPT}" "$(date '+%F %T')" > "${DIR}/exported/CURRENT_CANDIDATE.txt"

# --- 2) 助走スケール
dep="${DIR}/params/deploy.yaml"
if ! grep -q keyboard_vel_scale "${dep}"; then
  cp -n "${dep}" "${dep}.bak_before_vel_scale"
  python3 - "${dep}" << 'PY'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace("  base_velocity:\n    ranges:","  base_velocity:\n    keyboard_vel_scale: 1.0\n    ranges:",1)
open(p,"w").write(s)
PY
  echo "  deploy.yaml に keyboard_vel_scale: 1.0 を追記"
fi

# --- 3) symlink
ln -sfn "./${RUN}/exported" "${EXPROOT}/exported"
ln -sfn "./${RUN}/params"   "${EXPROOT}/params"
echo "=== ${ROBOT}: ${RUN}/${CKPT} をロード ==="
echo "  exported -> $(readlink "${EXPROOT}/exported")"

# --- 4) 検算
/home/tanaka/isaacsim/env_isaaclab/bin/python - "${EXPROOT}/exported/policy.onnx" "${EXPROOT}/params/deploy.yaml" "${DIR}/${CKPT}.pt" << 'PY'
import sys, onnx, yaml
g=onnx.load(sys.argv[1]).graph
dim=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
y=yaml.safe_load(open(sys.argv[2])); obs=y["observations"]
tot=sum(len(v["scale"])*(v.get("history_length",1) or 1) for v in obs.values() if isinstance(v,dict))
print(f"  ONNX入力 {dim} / deploy.yaml 観測合計 {tot} -> {'OK' if dim[-1]==tot else '*** 不一致、動かすな ***'}")
print("  keyboard_velocity_commands:", "あり" if "keyboard_velocity_commands" in obs else "*** 無し(キーが効かない) ***")
r=y["commands"]["base_velocity"]["ranges"]["lin_vel_x"]
sc=y["commands"]["base_velocity"].get("keyboard_vel_scale",0.5)
print(f"  助走: f = {sc*r[1]:.2f} m/s / b = {sc*r[0]:.2f} m/s")
# ONNX の重みが本当にその .pt かを照合する(同名で中身の違う ONNX の事故を2回やっている)
import numpy as np, torch
from onnx import numpy_helper
inits={i.name: numpy_helper.to_array(i) for i in g.initializer}
sd=torch.load(sys.argv[3], map_location="cpu")["model_state_dict"]
for key in ("actor.0.weight","estimator.0.weight"):
    ref=sd[key].numpy()
    hit=[n for n,a in inits.items() if a.shape==ref.shape and np.allclose(a,ref,atol=1e-6)]
    print(f"  {key} {ref.shape} -> {'MATCH' if hit else '*** ckpt と一致しない、動かすな ***'}")
PY
