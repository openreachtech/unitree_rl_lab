#!/usr/bin/env bash
# Anaguma フェーズ2 — 助走帯を U(2.8, 3.3) に戻し、`model_2400` から継続する。
#
# フェーズ1 (`2026-09-14_20-00-36`、助走 U(0.0,3.3)・apexゲート(0.5,1.5)) で **Anaguma が初めて
# 自力で踏み切った**: unaided_real_jump 0.726、補助が 0 になった iter 1500 以降も維持。
# mujoco 実測では `model_2400` が **学習で一度も踏み切っていない助走 2.8 指令(実測2.59 m/s)で
# 飛距離 0.987m・跳躍95%・着地78%** を出しており、速い助走に載る能力は既にある。
# よってフェーズ2は新規学習ではなく継続。変更は 3 点だけ:
#
#   1. base_velocity.lin_vel_x  U(0.0,3.3) -> U(2.8,3.3)
#   2. jump_takeoff_apex の approach_lo/hi  (0.5,1.5) -> (2.0,2.5)  ＝ Go2 と同じゲートに復帰
#   3. initial_assist_scale 0.50 -> 0.0   踏切はもう自力でできるので教師を外す
#      (試行2の「走れる重みに補助を当てると打ち消される」を繰り返さないため)
#
# dense 報酬の m*g 正規化 (normalize_by_weight=True) と early_weight 0.25 はフェーズ1のまま。
#
# 80 iter のスモーク(smoke_anaguma_phase2.log)では unaided_real_jump 0.55 -> 0.73 -> 0.70 で崩れず、
# unaided_trigger_speed が 1.11 -> 1.79 -> 2.14 と上がり、apex の収入も 0.069 -> 0.121 に増えた。
#
# 判定は `unaided_real_jump` と `unaided_trigger_speed`(2.8 帯に乗るか)、その後 mujoco。
# `--resume` は tot_iter = 再開iter + max_iterations なので 2400 + 3000 = 5400 まで回る。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
cd "${REPO}"
export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}"
exec /home/tanaka/isaacsim/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Anaguma-LongJump-v1 --headless --num_envs 2048 --max_iterations 3000 \
  --resume --load_run ${START_RUN:-2026-09-14_22-39-08} --checkpoint ${START_CKPT:-model_3200}.pt \
  --deploy-keyboard-commands
