#!/usr/bin/env bash
# Go2 飛距離重視ポリシーの自律ループ用 run スクリプト（2026-09-15〜）。
#
# 起点と iter 数は環境変数で渡す:
#   START_RUN=<run dir name> START_CKPT=model_NNNNN ITERS=3000 bash scripts/run_go2_loop_cycle.sh
#
# 既定の起点は `2026-09-14_17-02-52/model_10899`（Loop 10 起点のやり直し系列、
# mujoco 40試行 @vx2.8 で 1.354m / 着地28% / 成立100%）。据え置きベスト
# `2026-09-13_14-19-07/model_29200`（1.929m / 着地12% / 成立85%）より短いが着地が2.3倍良く、
# 「飛距離重視」で詰めるならこちらの系列を伸ばすほうが筋が良い、と 2026-09-14 に判断した。
#
# SEED=<N> を渡すと乱数seedを変えられる（既定は agent_cfg の 42 固定。同じ起点・同じ設定で
# 回すと**ビット単位で同じ run になる**ことをサイクル12で確認済み＝引き直しにはseed変更が必須）。
#
# **変更してよいのは報酬関数のみ**（ユーザー指示、2026-09-15）。助走帯・assist・env数・
# iter数などタスク設定は触らない。`--resume` は tot_iter = 再開iter + ITERS。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
cd "${REPO}"
export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}"
exec /home/tanaka/isaacsim/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Unitree-Go2-LongJump-v1 --headless --num_envs 2048 --max_iterations "${ITERS:-3000}" \
  --resume --load_run "${START_RUN:-2026-09-14_17-02-52}" --checkpoint "${START_CKPT:-model_10899}.pt" \
  ${SEED:+--seed ${SEED}} \
  --deploy-keyboard-commands
