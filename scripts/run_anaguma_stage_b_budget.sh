#!/usr/bin/env bash
# Anaguma Stage B, iter 0 から。2026-09-14 夜の収支修正版（試行6）。
#
# 起点・env数・iter数は試行4 (2026-09-14_17-02-57) と完全に同じにしてある。
# 違うのは報酬側の2点だけで、それ以外を動かさないことが今回の run の目的:
#
#   1. jump_dense_reward を機体重量 m*g で正規化（normalize_by_weight=True）＋
#      curriculum の early_weight 2.5 -> 0.25。試行5(2026-09-14_19-56-07、iter48で打ち切り)は
#      weight だけ下げたが -1.464 対 飴 +0.297 でまだ赤字だった。単位から出して -0.002 になった。
#      (-std(接地力)[N] は報酬セットで唯一 質量正規化されていない量。Anaguma は Go2 の
#       1.62 倍の質量で、跳んだ瞬間の支払いが -3.478 / Go2 は -0.0064 だった)
#   2. jump_takeoff_apex の approach_lo/hi 2.0/2.5 -> 0.5/1.5
#      (フェーズ1の助走帯 U(0.0,3.3) に対しゲートが 2.0 m/s 未満に一切払わず、
#       重み50 のこの項の収入が全区間 0.0000 だった)
#
# 判定は `Metrics/jump_command/unaided_real_jump_fraction`（補助なしホールドアウト25%）。
# 試行1〜4 はこれが全 iter 0.0000。ここが立てばフェーズ2（助走帯を U(2.8,3.3) へ戻し、
# approach_lo/hi も (2.0,2.5) へ戻す）に進む。
#
# 80 iter のスモーク(smoke_anaguma_budget_fix80.log)では飴 +0.203 -> +0.661、
# dense -3.478 -> -0.515、収支はほぼ均衡まで戻った。ただし 80 iter の中では
# real_jump が 0.96 -> 0.31 と下がる形も見えているので、そこは本番で確認する。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
cd "${REPO}"
export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}"
exec /home/tanaka/isaacsim/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Anaguma-LongJump-v1 --headless --num_envs 2048 --max_iterations 3000 \
  --resume --load_run 2026-09-14_15-29-28_net2net_anaguma_scratch --checkpoint model_0.pt \
  --deploy-keyboard-commands
