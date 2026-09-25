#!/usr/bin/env bash
# mujoco(C++/DDS 経路)を2窓で立ち上げる。go2 / anaguma 共通。
#   DISPLAY=:1 bash scripts/mujoco_launch.sh anaguma
# 事前に mujoco_load_policy.sh でポリシーを差し替えておくこと。
set -uo pipefail
ROBOT=${1:?usage: mujoco_launch.sh <go2|anaguma>}
REPO=/home/tanaka/isaacsim/unitree_rl_lab
export DISPLAY=${DISPLAY:-:1}
case "${ROBOT}" in
  anaguma) CTRLDIR=${REPO}/deploy/robots/anaguma/build; CTRL=anaguma_ctrl; EXPROOT=${REPO}/logs/rsl_rl/anaguma_longjump_v1 ;;
  go2)     CTRLDIR=${REPO}/deploy/robots/go2/build;     CTRL=go2_ctrl;     EXPROOT=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1 ;;
  *) echo "robot は go2 か anaguma"; exit 1 ;;
esac
# 古いプロセスを落とす(DDS の lowcmd チャンネルは1つしか持てない。2つ動くと指令が競合する)
pkill -f "unitree_mujoco -r " 2>/dev/null || true
pkill -f "go2_ctrl --network" 2>/dev/null || true
pkill -f "anaguma_ctrl --network" 2>/dev/null || true
sleep 3
cd /home/tanaka/isaacsim/unitree_mujoco/simulate/build
setsid nohup ./unitree_mujoco -r "${ROBOT}" -s scene_flat.xml > /tmp/mj_${ROBOT}.log 2>&1 < /dev/null & disown
sleep 10
NOW=$(cat "${EXPROOT}/exported/CURRENT_CANDIDATE.txt" 2>/dev/null | head -1)
setsid nohup gnome-terminal --title="${CTRL}  ${NOW}  /  1→Enter→b→(4〜5秒)→j  /  転んだら mujoco窓で Backspace" \
  --geometry=118x30 -- bash -c "cd ${CTRLDIR} && ./${CTRL} --network lo; echo; echo '=== 終了 ==='; read" \
  > /tmp/ctrl_${ROBOT}.log 2>&1 < /dev/null & disown
sleep 6
echo "=== 起動 ==="; ps -eo pid,args | grep -E "[u]nitree_mujoco -r ${ROBOT}|[${CTRL:0:1}]${CTRL:1} --network" | grep -v bash
echo "ロード中: ${NOW}"
