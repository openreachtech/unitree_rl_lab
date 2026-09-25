#!/usr/bin/env bash
# go2_ctrl に読ませる候補を差し替える。
#   bash scripts/mujoco_load_candidate.sh G4a
#
# やること3つ:
#   1) eval_go2_cycle4/<tag>_*.onnx を その run の exported/policy.onnx に上書き
#   2) logs/rsl_rl/unitree_go2_longjump_v1/ 直下の exported/params symlink をその run へ張替え
#      (go2_ctrl は直下に exported/ があると サブdir を探索しない。param.h:95)
#   3) ONNX 入力次元 と deploy.yaml の観測合計が一致するか検算（不一致は例外を出さずに壊れる）
set -euo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
EXPROOT=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1
SRC=${REPO}/eval_go2_cycle4
tag=${1:?usage: mujoco_load_candidate.sh <G4a|G4b|G4c|G4x|REF>}

onnx=$(ls "${SRC}/${tag}"_*.onnx 2>/dev/null | head -1)
[ -n "${onnx}" ] || { echo "no onnx for tag ${tag} in ${SRC}"; exit 1; }
base=$(basename "${onnx}" .onnx)                 # <tag>_<run>_<ckpt>
rest=${base#${tag}_}; run=${rest%_model_*}; ckpt=model_${rest##*_model_}

cp -f "${onnx}" "${EXPROOT}/${run}/exported/policy.onnx"
rm -f "${EXPROOT}/${run}/exported/policy.pt"     # 学習時の残骸。go2_ctrl は読まないが紛らわしい
printf '%s  %s/%s  (loaded %s)\n' "${tag}" "${run}" "${ckpt}" "$(date '+%F %T')" \
  > "${EXPROOT}/${run}/exported/CURRENT_CANDIDATE.txt"
ln -sfn "./${run}/exported" "${EXPROOT}/exported"
ln -sfn "./${run}/params"   "${EXPROOT}/params"

echo "=== loaded: ${tag} = ${run}/${ckpt} ==="
echo "  exported -> $(readlink "${EXPROOT}/exported")"
echo "  params   -> $(readlink "${EXPROOT}/params")"
# 4) 助走コマンドを学習時の帯そのままにする。keyboard_velocity_commands は
#    vel_scale(既定 0.5) x commands.base_velocity.ranges で速度を作るので、既定のままだと
#    f = 0.5 x 3.3 = 1.65 m/s しか出ず、学習帯 U(2.8, 3.3) の半分の動作点で測ることになる。
#    (State_RLBase.cpp:33 / :58, 値は deploy.yaml の commands.base_velocity から読む)
dep="${EXPROOT}/${run}/params/deploy.yaml"
if ! grep -q "keyboard_vel_scale" "${dep}"; then
  cp -n "${dep}" "${dep}.bak_before_vel_scale"
  python3 - "${dep}" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = s.replace("  base_velocity:\n    ranges:", "  base_velocity:\n    keyboard_vel_scale: 1.0\n    ranges:", 1)
open(p, "w").write(s)
PYEOF
  echo "  deploy.yaml: keyboard_vel_scale 1.0 を追記 (backup: deploy.yaml.bak_before_vel_scale)"
fi
grep -n "keyboard_vel_scale" "${dep}" | sed 's/^/  /'
echo "  -> f = 上限 $(grep -A3 'ranges:' "${dep}" | grep lin_vel_x | sed 's/.*\[//;s/\]//' | cut -d, -f2) m/s  /  b = 下限 $(grep -A3 'ranges:' "${dep}" | grep lin_vel_x | sed 's/.*\[//;s/\]//' | cut -d, -f1) m/s"

/home/tanaka/isaacsim/env_isaaclab/bin/python - "${EXPROOT}/exported/policy.onnx" "${EXPROOT}/params/deploy.yaml" <<'PY'
import sys, onnx, yaml
g = onnx.load(sys.argv[1]).graph
dim = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
obs = yaml.safe_load(open(sys.argv[2]))["observations"]
tot = sum(len(v["scale"]) * (v.get("history_length", 1) or 1) for v in obs.values() if isinstance(v, dict))
kb = "keyboard_velocity_commands" in obs
print(f"  ONNX input {dim}  /  deploy.yaml obs 合計 {tot}  -> {'OK' if dim[-1] == tot else '*** 不一致 ***'}")
print(f"  keyboard_velocity_commands: {'あり' if kb else '*** 無し（キーボードが効かない）***'}")
PY
