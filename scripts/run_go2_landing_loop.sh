#!/usr/bin/env bash
# =============================================================================
# Go2 走り幅跳び: **着地率だけ**を目的にした自律ループ（2026-09-16、ユーザー指示）
# =============================================================================
#
# 目的: 飛距離は十分（遅延ゼロ条件で 1.338 m）。以後は **着地成功率のみ**を上げる。
#
# 【このループの絶対規則】
#   1. **推論（こうだろう）で施策を打たない。** 試すレバーは、このプロジェクトで
#      すでに「実測」された2つに限定する（下の ARM 一覧）。新しい報酬項の発明、
#      未測定のパラメータいじりはしない。
#   2. **採否は測定だけで決める。** 各アームは同一プロトコル（同じ起点・同じ iter 数・
#      同じ評価）で回し、着地率が現行ベストを上回ったときだけ採用する。
#   3. **評価は必ず遅延 1 ステップ（20 ms）で行う。** 2026-09-16 の実測で、遅延ゼロの
#      数字は実機も C++ 経路も予測しないと分かっている（着地 48% -> 5%）。
#   4. **飛距離のガード**: 現行ベストの 70% を下回る個体は、着地率が高くても採用しない
#      （跳ばないポリシーが着地率で勝つのを防ぐため。評価側でも takeoff 必須）。
#
# 【事前登録アーム（実測済みのレバーのみ）】
#   A0 対照   … 設定変更なし。seed だけ変えて引き直す（他アームの比較基準）
#   A1 jump_pitch_rate = -12.0 / max_rate 3.0
#        実測(Loop 40, 2026-09-15): 着地 28% -> 68〜78%（プロジェクト記録）。代償は飛距離 -26%。
#        当時のコメント「着地を目的にするなら最初にここへ戻る」に従う。
#   A2 助走帯 U(2.8,3.3) -> U(3.3,3.8)
#        実測(2026-09-16 助走レバー検証): 速い帯は着地が桁で良い（2% -> 28%、コマンド4.0で60%）。
#        代償は飛距離 -21%。
#   A3 A1+A2 … **A1 と A2 が両方とも対照を上回ったときだけ**実行する。
#
# 学習条件は固定: SIM_DT=0.002 / ACTION_DELAY_STEPS=0,2 / 2048 env / 300 iter/サイクル。
#
# **ITERS は「追加 iter 数」**（絶対値ではない）。rsl_rl の --resume は tot_iter = 再開iter + 指定値。
# 2026-09-16 に絶対値を渡し、1200 のつもりが 13900 iter 回してしまった（4時間浪費）。
# 停止: 締切（HOURS、既定8時間）。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
EXP=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1
SCENE=/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/go2/scene_flat.xml
PY=/home/tanaka/isaacsim/env_isaaclab/bin/python
MJPY=/home/tanaka/isaacsim/anaguma/venv/bin/python
OUT=${REPO}/eval_landing_loop_20260916
STATE=${REPO}/.go2_landing_loop_state
REPORT=${REPO}/docs/go2_landing_loop_20260916.md
cd "${REPO}"; export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
export SIM_DT=0.002 ACTION_DELAY_STEPS=0,2
DEADLINE=$(( $(date +%s) + ${HOURS:-8} * 3600 ))
ITERS_PER_CYCLE=300
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }
note() { echo "$*" >> "${REPORT}"; }

# --- 先行ジョブ（遅延DRの学習・評価）の終了を待つ ----------------------------
while pgrep -f "[r]un_go2_delay_dr.sh" > /dev/null || pgrep -f "[e]val_delay_dr_sweep.sh" > /dev/null; do sleep 60; done
say "先行の遅延DRジョブが終了。着地ループを開始する。"

# --- 起点: 遅延DR run の中で「遅延1での着地率」が最良だった ckpt -------------
# スイープ評価が残した最良を最優先で使う（無ければ doc を解析、それも無ければ最終 ckpt）
if [ -s "${REPO}/.go2_delay_dr_best" ]; then
  # shellcheck disable=SC1090
  . "${REPO}/.go2_delay_dr_best"
  say "起点をスイープ結果から取得: ${BEST_RUN}/${BEST_CKPT}（着地 ${BEST_LAND}% / 飛距離 ${BEST_DIST} m）"
fi
DR_DOC=${REPO}/docs/go2_delay_dr_20260916.md
DR_RUN=${BEST_RUN:-$(grep -oP '^run: `\K[^`]+' "${DR_DOC}" 2>/dev/null | head -1)}
if [ -n "${BEST_CKPT:-}" ]; then SKIP_DOC_PARSE=1; fi
[ "${SKIP_DOC_PARSE:-0}" = "1" ] || read -r BEST_CKPT BEST_LAND BEST_DIST <<< "$(
  awk -F'|' '/^\| `model_/ {
      gsub(/[` ]/,"",$2); land=$5; dist=$3; gsub(/[% ]/,"",land); gsub(/[m ]/,"",dist);
      if (land+0 > bl+0 || (land+0 == bl+0 && dist+0 > bd+0)) { bc=$2; bl=land; bd=dist }
    } END { printf "%s %s %s\n", (bc==""?"NONE":bc), (bl==""?0:bl), (bd==""?0:bd) }' "${DR_DOC}" 2>/dev/null)"
if [ "${BEST_CKPT}" = "NONE" ] || [ -z "${DR_RUN}" ]; then
  DR_RUN=$(ls -1t "${EXP}" | grep '^2026-' | head -1); BEST_CKPT=$(ls -1 "${EXP}/${DR_RUN}"/model_*.pt | sort -t_ -k2 -n | tail -1 | xargs basename | sed 's/.pt//')
  BEST_LAND=0; BEST_DIST=0
  say "遅延DRの表を読めなかった。フォールバック: ${DR_RUN}/${BEST_CKPT}"
fi
BEST_RUN=${DR_RUN}
say "起点: ${BEST_RUN}/${BEST_CKPT}（遅延1で着地 ${BEST_LAND}% / 飛距離 ${BEST_DIST} m）"

note ""
note "# Go2 着地率ループ（2026-09-16〜）"
note ""
note "**目的は着地率のみ**（飛距離は十分という判断、ユーザー指示 2026-09-16）。"
note "**評価は全て \`--action-delay 1\`（20 ms）・mujoco 40試行・助走コマンド 2.8 m/s。**"
note "遅延ゼロの数字は実機も C++ 経路も予測しないため使わない。"
note ""
note "試すレバーは**このプロジェクトで実測済みの2つだけ**に事前登録した（推論で施策を足さない）:"
note "A1 \`jump_pitch_rate\` −12.0（実測: 着地 28→78%、飛距離 −26%）／"
note "A2 助走帯 U(3.3,3.8)（実測: 着地 2→28%、コマンド4.0で60%、飛距離 −21%）。"
note ""
note "起点: \`${BEST_RUN}/${BEST_CKPT}\`（遅延1で着地 ${BEST_LAND}% / 飛距離 ${BEST_DIST} m）"
note ""
note "| # | アーム | seed | 評価vx | 最良 ckpt | 着地(遅延1) | 飛距離(遅延1) | 成立 | 判定 |"
note "|---|---|---|---|---|---|---|---|---|"

# --- 評価 ---------------------------------------------------------------------
eval_one() {  # dir ckpt tag vx -> "距離 成立 着地"
  local dir=$1 ckpt=$2 tag=$3 vx=$4
  local onnx="${OUT}/${tag}.onnx"
  if [ ! -s "${onnx}" ]; then
    rm -f "${dir}/exported/policy.onnx"
    ${PY} scripts/rsl_rl/play.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 1 \
      --checkpoint "${dir}/${ckpt}.pt" > "${OUT}/export_${tag}.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 90); do sleep 5; [ -s "${dir}/exported/policy.onnx" ] && break; done
    sleep 5; kill -9 "${pid}" 2>/dev/null; sleep 3
    [ -s "${dir}/exported/policy.onnx" ] || { echo "0 0 0"; return 1; }
    cp "${dir}/exported/policy.onnx" "${onnx}"
  fi
  ${MJPY} scripts/mujoco_jump_eval.py --run "${dir}" --policy "${onnx}" --scene "${SCENE}" \
    --trials 40 --vx "${vx}" --action-delay 1 > "${OUT}/mj_${tag}.log" 2>&1
  awk '/飛距離 平均/{d=$3} /跳躍成立/{for(i=1;i<=NF;i++) if($i ~ /^\(/){gsub(/[()%]/,"",$i); if(j=="") {j=$i; continue}; l=$i}} END{printf "%s %s %s\n", (d==""?0:d), (j==""?0:j), (l==""?0:l)}' "${OUT}/mj_${tag}.log"
}

CYCLE=0
# --- 1サイクル ----------------------------------------------------------------
run_cycle() {  # arm_name seed eval_vx extra_env...  -> グローバル C_LAND C_DIST C_JUMP C_CKPT C_RUN
  #
  # eval_vx は「そのアームが学習した助走帯の下限」＝実機で `b` キーが出す速度。
  # A2 は帯ごと上げるアームなので、2.8 で測ると学習分布の外で測ることになり、
  # アームの効果ではなく分布外性能を測ってしまう。帯の下限で測るのが同じ条件。
  local arm=$1 seed=$2 vx=$3; shift 3
  C_VX=${vx}
  CYCLE=$((CYCLE + 1))
  local start_it=${BEST_CKPT#model_}
  say "サイクル${CYCLE} [${arm}] seed=${seed} 評価vx=${vx} 起点=${BEST_RUN}/${BEST_CKPT} 追加env: $*"
  env "$@" START_RUN="${BEST_RUN}" START_CKPT="${BEST_CKPT}" \
      ITERS="${ITERS_PER_CYCLE}" SEED="${seed}" \
      bash scripts/run_go2_loop_cycle.sh > "${REPO}/go2_landing_c${CYCLE}.log" 2>&1
  C_RUN=$(ls -1t "${EXP}" | grep '^2026-' | head -1)
  C_LAND=0; C_DIST=0; C_JUMP=0; C_CKPT=""
  for off in 100 150 200 250 300; do
    local it=$((start_it + off))
    [ -s "${EXP}/${C_RUN}/model_${it}.pt" ] || continue
    read -r d j l <<< "$(eval_one "${EXP}/${C_RUN}" "model_${it}" "c${CYCLE}_${it}" "${vx}")"
    say "  model_${it}: 着地 ${l}%  飛距離 ${d} m  成立 ${j}%"
    if awk -v a="${l}" -v b="${C_LAND}" 'BEGIN{exit !(a>b)}'; then
      C_LAND=${l}; C_DIST=${d}; C_JUMP=${j}; C_CKPT="model_${it}"
    fi
  done
}

# --- 採用判定（測定のみ） ------------------------------------------------------
adopt_if_better() {  # arm seed
  local arm=$1 seed=$2 floor verdict
  floor=$(awk -v b="${BEST_DIST}" 'BEGIN{printf "%.3f", b*0.7}')
  if awk -v l="${C_LAND}" -v b="${BEST_LAND}" 'BEGIN{exit !(l>b)}' && \
     awk -v d="${C_DIST}" -v f="${floor}" 'BEGIN{exit !(d>=f)}'; then
    BEST_RUN=${C_RUN}; BEST_CKPT=${C_CKPT}; BEST_LAND=${C_LAND}; BEST_DIST=${C_DIST}
    verdict="**採用（着地 ${C_LAND}%）**"
    printf 'BEST_RUN=%s\nBEST_CKPT=%s\nBEST_LAND=%s\nBEST_DIST=%s\nARM=%s\nCYCLE=%s\n' \
      "${BEST_RUN}" "${BEST_CKPT}" "${BEST_LAND}" "${BEST_DIST}" "${arm}" "${CYCLE}" > "${STATE}"
  elif awk -v d="${C_DIST}" -v f="${floor}" 'BEGIN{exit !(d<f)}'; then
    verdict="不採用（飛距離 ${C_DIST} < ガード ${floor}）"
  else
    verdict="不採用（ベスト ${BEST_LAND}% のまま）"
  fi
  note "| ${CYCLE} | ${arm} | ${seed} | ${C_VX} | \`${C_CKPT}\` | ${C_LAND}% | ${C_DIST} m | ${C_JUMP}% | ${verdict} |"
  say "サイクル${CYCLE} ${verdict}"
}

alive() { [ "$(date +%s)" -lt "${DEADLINE}" ]; }

# --- アームを順に測る ----------------------------------------------------------
A0_BEST=0; A1_BEST=0; A2_BEST=0
for s in 401; do alive || break; run_cycle "A0 対照" ${s} 2.8 DUMMY=1; adopt_if_better "A0 対照" ${s}
  awk -v a="${C_LAND}" -v b="${A0_BEST}" 'BEGIN{exit !(a>b)}' && A0_BEST=${C_LAND}; done
for s in 411 412; do alive || break; run_cycle "A1 pitch-12" ${s} 2.8 JUMP_PITCH_RATE_W=-12.0; adopt_if_better "A1 pitch-12" ${s}
  awk -v a="${C_LAND}" -v b="${A1_BEST}" 'BEGIN{exit !(a>b)}' && A1_BEST=${C_LAND}; done
for s in 421 422; do alive || break; run_cycle "A2 助走3.3-3.8" ${s} 3.3 APPROACH_SPEED_MIN_MS=3.3 APPROACH_SPEED_MS=3.8; adopt_if_better "A2 助走3.3-3.8" ${s}
  awk -v a="${C_LAND}" -v b="${A2_BEST}" 'BEGIN{exit !(a>b)}' && A2_BEST=${C_LAND}; done

# A3 は「A1 と A2 が両方とも対照を上回った」ときだけ（測定に基づく条件）
if awk -v a="${A1_BEST}" -v b="${A0_BEST}" 'BEGIN{exit !(a>b)}' && awk -v a="${A2_BEST}" -v b="${A0_BEST}" 'BEGIN{exit !(a>b)}'; then
  note "| — | A3 実行条件 | — | — | — | — | — | — | A1(${A1_BEST}%) と A2(${A2_BEST}%) が両方とも対照(${A0_BEST}%)超え → 併用を試す |"
  for s in 431 432; do alive || break; run_cycle "A3 併用" ${s} 3.3 JUMP_PITCH_RATE_W=-12.0 APPROACH_SPEED_MIN_MS=3.3 APPROACH_SPEED_MS=3.8; adopt_if_better "A3 併用" ${s}; done
else
  note "| — | A3 実行条件 | — | — | — | — | — | — | 満たさず（A1 ${A1_BEST}% / A2 ${A2_BEST}% / 対照 ${A0_BEST}%）→ 併用は試さない |"
fi

# --- 勝ったアームの設定で、締切まで seed を変えて積む -------------------------
WIN_ENV=(DUMMY=1); WIN_NAME="A0 対照"; WIN_VX=2.8
if awk -v a="${A1_BEST}" -v b="${A0_BEST}" 'BEGIN{exit !(a>b)}'; then WIN_ENV=(JUMP_PITCH_RATE_W=-12.0); WIN_NAME="A1 pitch-12"; WIN_VX=2.8; fi
if awk -v a="${A2_BEST}" -v b="${A1_BEST}" 'BEGIN{exit !(a>b)}' && awk -v a="${A2_BEST}" -v b="${A0_BEST}" 'BEGIN{exit !(a>b)}'; then
  WIN_ENV=(APPROACH_SPEED_MIN_MS=3.3 APPROACH_SPEED_MS=3.8); WIN_NAME="A2 助走3.3-3.8"; WIN_VX=3.3; fi
note ""
note "**測定で勝ったアーム: ${WIN_NAME}**（対照 ${A0_BEST}% / A1 ${A1_BEST}% / A2 ${A2_BEST}%）。以降はこの設定で seed を変えて積む。"
note ""
note "| # | アーム | seed | 評価vx | 最良 ckpt | 着地(遅延1) | 飛距離(遅延1) | 成立 | 判定 |"
note "|---|---|---|---|---|---|---|---|---|"
s=500
while alive; do
  s=$((s + 1))
  run_cycle "${WIN_NAME}" ${s} "${WIN_VX}" "${WIN_ENV[@]}"
  adopt_if_better "${WIN_NAME}" ${s}
done

note ""
note "## 最終結果"
note ""
note "**着地率 ${BEST_LAND}%（遅延1ステップ・40試行）/ 飛距離 ${BEST_DIST} m: \`${BEST_RUN}/${BEST_CKPT}\`**"
note ""
note "終了 $(ts)。"
say "ループ終了: ベスト ${BEST_RUN}/${BEST_CKPT} 着地 ${BEST_LAND}% 飛距離 ${BEST_DIST} m"
