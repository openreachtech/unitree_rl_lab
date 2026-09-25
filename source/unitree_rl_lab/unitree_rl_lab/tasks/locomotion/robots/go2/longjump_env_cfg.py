"""Stage B of the running long jump task ("Unitree-Go2-LongJump-v1").

Jump-integrated env on top of the base velocity-tracking task: adds the
binary ``jump_command`` (with the EFGCL external-force assist), the Explicit
Estimator's regression-target observation group, the Dense/Sparse Jump Reward
pair, a jump-aware orientation termination, and the jump-target-speed and
assist-decay curricula. See:
- リファレンス_ExplicitEstimator実装仕様.md (Estimator + Net2Net design, PPOEE/ActorCriticEE plan)
- リファレンス_キーボード操作とNet2Net設計の下調べ.md (Stage A/B split, task naming)
- プロジェクト_走り幅跳び理論値計算.md (v_target curriculum ceiling rationale)
- プロジェクト_StageB崩壊の原因分析.md (the 2026-09-02 rework: why the previous run
  collapsed to "fall over immediately" and what each change here fixes)

Everything about the reward budget, the corrected actuator model and the domain
randomisation is inherited from Stage A's config (RewardsCfgLongJumpBase /
EventCfgLongJump / GO2_CORRECTED_ACTUATOR_CFG) so the two stages cannot drift apart.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.longjump_base_env_cfg import (
    CommandsCfgLongJumpBase,
    RewardsCfgLongJumpBase,
    RobotEnvCfgLongJumpBase,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    CurriculumCfg,
    ObservationsCfg,
    TerminationsCfg,
)


@configclass
class CommandsCfgLongJump(CommandsCfgLongJumpBase):
    """Stage A's forward-only velocity command (inherited), plus the jump command.

    2026-09-02 jump_command rework, all three numbers changed for a measured reason
    (see JumpCommand's own docstring for the mechanism and
    プロジェクト_StageB崩壊の原因分析.md for the run data):

    - ``min_air_time_s`` 0.05 -> 0.35. At 0.05 s (2.5 control steps) a standing robot
      satisfied the landing condition almost immediately after the trigger, so the
      command cleared itself before any jump could be attempted. Across the whole of
      the previous two runs ``jumping_fraction`` never exceeded 4% and
      ``jump_vel_target_levels`` never moved off its initial value -- the policy was
      never actually asked to jump. A real crouch-and-push-off takes 0.2-0.4 s.
    - ``max_jump_duration_s`` 1.5 (new). Hard, pose-independent timeout, so a fallen
      robot cannot hold the command at 1 forever (which also used to disable the
      orientation termination).
    - EFGCL assist enabled. The assist values are tak's proven Go2-Jump numbers
      (crouch 150 N for 0.12 s, then a 0.12 s ramp into the launch), with the launch
      force sized here for a 0.20 m apex -- the apex
      [[reference_paper_impedance_matching_running_jump]] measured for its forward
      running jump (Table III: 24.4 cm at a 2 m/s approach).
    """

    jump_command = mdp.JumpCommandCfg(
        resampling_time_range=(3.0, 5.0),
        jump_probability=0.5,
        min_air_time_s=0.35,
        # 2026-09-03 (v5): 1.5 -> 2.2 s. Airborne time reached 0.6-0.69 s in v4, and the
        # landing condition (all four feet down AND upright) then routinely failed to
        # complete before the timeout: success_rate sat at ~15% while
        # real_jump_fraction sat at ~90%. Because jump_distance is only paid at a
        # scoring landing, that timeout was also silently zeroing the one term that
        # rewards the task objective. A longer window lets the descent finish.
        max_jump_duration_s=2.2,
        landed_upright_angle=0.4,
        # 2026-09-06 (Loop 15): 2 -> 1. The Loop 14 diagnostics say the "landing collapse"
        # of Loops 12-14 was never a landing failure at all -- it was a DETECTION failure.
        # At the collapse, unaided_timeout_fraction is 0.73-0.97 (windows close on the
        # 2.2 s timeout, not on a landing) while unaided_upright_at_close is 0.93-0.99 and
        # bad_orientation is 0.01-0.06: the robot is upright and not falling, it simply
        # never presents two feet at the same instant. A Go2 running out of a jump at
        # 1.3 m/s puts its feet down one at a time; Loop 5 already had to relax this from
        # 4 to 2 for the same reason, and at this speed 2 is still too many.
        #
        # This is not only a metric: jump_command is cleared by ``landed | timed_out``, so
        # a window that times out keeps the policy in jump mode for 2.2 s instead of ~0.9 s
        # -- while the deployed controller drops it after a fixed 0.7 s. The training and
        # deploy distributions had been drifting apart at exactly the moment of landing.
        min_feet_down_for_landing=1,
        # --- EFGCL assist ---
        # 2026-09-05 (Loop 10): 1.0 -> 0.0, i.e. the whole run is unaided.
        #
        # Two measurements, both from the Loop 9 post-mortem:
        #
        # 1. The teacher is now weaker than the student. The launch force is sized to
        #    impart v0 = sqrt(2*g*assist_apex_height_m) = 1.53 m/s by itself, for a
        #    0.12 m apex. v8's model_7600 takes off unaided at v_z = 1.62 m/s for a
        #    0.119 m apex -- so the assist no longer demonstrates anything the policy
        #    cannot already do, it only perturbs the 75% of envs that receive it.
        # 2. Loop 9 spent 1100 of its 1500 iterations with the assist still fading, and
        #    the unaided numbers declined monotonically through the 400 that were left
        #    (distance 0.339 -> 0.317, height 0.111 -> 0.100). Three loops running, the
        #    best checkpoint has not been the final one. Starting at 0 makes every
        #    iteration and all 4096 envs train the thing that gets deployed.
        #
        # The assist stays configured (and jump_assist_decay stays in the curriculum) so
        # that raising this back above zero is the only edit needed if a later loop wants
        # a stronger teacher -- which is the EFGCL route, and the fallback if Loop 10's
        # instrumentation shows the energy is lost in the take-off stance rather than to
        # deceleration during the approach.
        # 2026-09-05 (Loop 12): 0.0 -> 1.0. The Loop 10 objection was that the teacher had
        # become weaker than the student -- a 0.12 m apex is v0 = 1.53 m/s against v8's
        # unaided 1.62 m/s. That is no longer true in either direction: v10 takes off at
        # v_z = 1.378 m/s, and the apex below is raised to 0.20 m = 1.98 m/s. The assist
        # now demonstrates a push-off the policy demonstrably cannot produce, which is
        # the only condition under which EFGCL has anything to teach.
        #
        # This is the one lever the doc's own Loop 11 post-mortem named: three loops of
        # reward reweighting left v_x/v_z at 1.6-1.7, so the split is set by the take-off
        # posture, and a posture is shown physically rather than paid for.
        # 2026-09-06 (Loop 13): 1.0 -> 0.0. The teacher has been overtaken again. It is
        # sized for a 0.10 m setting = 1.98 m/s nominal and a measured apex of 0.15-0.20 m;
        # the Loop 12 policy takes off unaided at v_z = 1.893 m/s for a 0.174 m apex. The
        # condition for EFGCL to have anything to teach (assist demonstrably above the
        # policy, and below the reward target so it cannot hand out full marks) leaves
        # almost no room between 0.174 and 0.28, so the whole run is unaided again.
        # 2026-09-14: 0.0 -> 1.0, for the from-scratch Stage B run. Every value above was
        # argued on a policy that already jumped -- Loop 13 switched the teacher off
        # because the student had overtaken it. A Stage B starting at iteration 0 has no
        # student: the whole jump reward set sits behind the real_jump gate, so with no
        # assist it pays exactly nothing and nothing produces the first take-off. That is
        # not a hypothesis, it is what the 2026-09-13 Anaguma run measured over 3000
        # iterations (real_jump 0.0000 throughout). The 0.10 apex setting below is left
        # alone: Go2 measured ~0.24 m from it, above what a fresh policy can do and below
        # the 0.30 m jump_height target, which is the condition EFGCL needs.
        # 2026-09-14 夕: 1.0 -> 0.0 に戻す。上の from-scratch 用の理屈はそのまま正しいが、
        # from-scratch 自体を取り下げたため（助走 U(2.8,3.3) でゼロから踏切を学ぶのは、
        # 過去に一度も成立していない条件だった）。起点は Loop 10 の model_9400 で、これは
        # 補助ゼロで real_jump 75% を出すポリシー——教師は要らないどころか、弱い教師は
        # 外乱にしかならない（Loop 13 がこの設定を 0 にしたときと同じ判断）。
        initial_assist_scale=0.0,
        crouch_assist_force=150.0,
        crouch_assist_duration_s=0.12,
        assist_delay_s=0.12,
        assist_ramp_s=0.12,
        assist_duration_s=0.10,
        # Sized for LESS than the reward's target_height (0.28 m) on purpose -- see the
        # field's docstring. In v4 these two numbers were equal at 0.20 m, so the assist
        # delivered the full rewarded apex by itself.
        # 2026-09-05 (Loop 12): 0.12 -> 0.10, and the number is now set by MEASUREMENT
        # rather than by the formula. The constraint is the v4 lesson -- the assisted apex
        # must stay below the reward's 0.28 m target, or the teacher hands out full marks
        # by itself and the policy learns nothing (in v4 the two were equal at 0.20 m and
        # 100% of the jump turned out to be the external force).
        #
        # Smoke-measured max_height_gain against this setting, all at scale 1.0:
        #     0.20 (pre-impulse-fix, effective 0.512)  ->  0.40-0.61 m   far over target
        #     0.20 (post-fix)                          ->  0.30-0.38 m   still over
        #     0.10                                     ->  ~0.24 m       under, with margin
        # The relation is not the formula's: the assist adds velocity to a push-off the
        # policy is already making, so the apexes compose as (sqrt(h_policy) + k*v0)^2.
        # Fitting the two points above gives apex ~= (0.259 + 0.725*sqrt(setting))^2.
        assist_apex_height_m=0.35,
        # A quarter of the envs never get the assist, so unaided_* metrics report what
        # the deployed policy will actually do.
        assist_holdout_fraction=0.25,
        debug_vis=False,
    )


@configclass
class PolicyCfgLongJump(ObservationsCfg.PolicyCfg):
    """Net2Net: the jump command bit plus its phase, appended after the inherited 45.

    ``jump_time`` (2026-09-02) is the second new input. A bare 0/1 bit told the policy
    nothing about *where in the jump it was*, so one feed-forward mapping had to serve
    both "load the legs now" and "you have been airborne 300 ms, prepare to land".
    tak's working Go2-Jump feeds the same quantity. Dataclass inheritance appends both
    after the parent's 45 terms, which is what longjump_net2net_transplant.py's column
    surgery assumes (base columns first, new columns zero-initialised).
    """

    jump_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "jump_command"})
    jump_time = ObsTerm(func=mdp.jump_time_encoding, params={"command_name": "jump_command"})


@configclass
class EstimatorTargetCfg(ObsGroup):
    """Explicit Estimator regression targets: ground truth, critic-side only.

    Never fed to the policy group. Phase-1 minimal label set agreed
    2026-08-27 (see リファレンス_ExplicitEstimator実装仕様.md): base linear
    velocity (3), per-link contact (17, reuses the existing
    ``contact_forces`` sensor's wildcard body coverage), foot height above
    the flat ground (4, no height-scanner needed since terrain is flat).
    """

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    link_contact = ObsTerm(func=mdp.link_contact_bool)
    foot_height = ObsTerm(func=mdp.foot_height)

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class ObservationsCfgLongJump(ObservationsCfg):
    policy: PolicyCfgLongJump = PolicyCfgLongJump()
    estimator_target: EstimatorTargetCfg = EstimatorTargetCfg()


@configclass
class RewardsCfgLongJump(RewardsCfgLongJumpBase):
    """Stage A's rebalanced budget (alive/upright + relaxed effort penalties) plus jumping.

    2026-09-02 changes to the two jump terms:

    - ``jump_sparse_reward`` weight 250 -> 25. At 250 this was the single largest
      positive term in the collapsed policy's return: reconstructing the numbers, a
      0.55 s episode spent falling over collected +15.4, against +21.6 for a *full
      18.5 s episode* of good velocity tracking. Its detection is now also strict
      (rising, upright, once per commanded jump -- see mdp.jump_sparse_reward), so a
      topple can no longer trigger it at all; the weight cut is the second line of
      defence and keeps one good lift-off worth roughly one second of the alive/upright
      baseline rather than an entire episode of running.
    - ``jump_dense_reward`` weight 2.5 -> 0.5, and it now only runs during the push-off
      (see mdp.jump_dense_reward). At 2.5 it was -0.558/s, the second largest penalty
      in the budget, and because std(F_foot) ~ 0 while lying on the ground it actively
      scored falling *higher* than pushing off.
    - ``std`` 1.0 kept: with v_target 1.5 m/s it means a lift-off within ~0.67 m/s of
      target still earns 80% of the term, which is what gates jump_vel_target_levels.
    """

    # Dense: the term that actually makes the robot leave the ground. Weight is set from
    # the trade-off the policy faces *inside* the jump window, where its alternative is
    # to keep running and collect track_lin_vel_xy (1.5 * ~0.77 = 1.15/s measured):
    #
    #   weight 1.5 -> a 0.20 m jump pays 1.50/s (wins) but a 0.10 m one pays 0.55/s (loses)
    #   weight 2.5 -> a 0.20 m jump pays 2.50/s (wins) and a 0.10 m one pays 0.92/s (~ties)
    #
    # At 1.5 the early, partial jumps the policy must pass through on the way to a real
    # one are worth less than simply running, i.e. the gradient points the wrong way
    # exactly where it matters. 2.5 keeps partial progress competitive. This is the same
    # reasoning tak used to raise non_target_rotation from -0.15 to -1.0 ("at -0.15 that
    # costs 0.067, two orders of magnitude under motion_progress's ~1.0, so there was
    # never a reason to stop twisting").
    # 2026-09-03 (v5): target 0.20 -> 0.28 m, scale 0.01 -> 0.02.
    #
    # Height is what the run is short of. tanaka's mujoco verdict on the v4 policy was
    # "the body does not float enough, so the landing fails" -- and the reward was the
    # reason: exp(-(h-0.20)^2/0.01) *peaks* at 0.20 m and falls away above it, so the
    # policy had a positive incentive not to exceed the number the assist was already
    # supplying. For a long jump more height is strictly better as long as the approach
    # speed holds, since distance = 2*v_x*v_z/g and v_z = sqrt(2*g*h): at v_x = 4.3 m/s,
    # h = 0.20 m gives 1.74 m and h = 0.28 m gives 2.06 m.
    #
    # The wider scale keeps the gradient reachable from where the policy starts -- with
    # 0.02 the term reads 0.10 at h = 0.13 m and 0.45 at h = 0.20 m, so the partial
    # jumps on the way up are still worth more than running (1.15/s at weight 2.5),
    # which is the trade-off the weight was chosen for in the first place.
    # 2026-09-06 (Loop 17): weight 2.5 -> 5.0 and the same 0.6 s payout bound, for the
    # same reason -- max_height_gain is a running maximum, so this term also paid more the
    # longer the window stayed open.
    jump_height = RewTerm(
        func=mdp.jump_height_progress,
        # 2026-09-06 (Loop 18): target 0.28 -> 0.38 m, weight 5.0 -> 7.5. The 0.28 m goal
        # -- the apex the reference paper measured for a forward running jump -- has been
        # REACHED: Loop 17 ends at a trunk rise of 0.288 m, which puts this Gaussian at
        # 0.997 of full. It is now a flat +5.0 with no gradient, the same death that took
        # jump_takeoff_speed (Loop 11) and was about to take jump_takeoff_apex (Loop 16).
        # 0.38 puts it back at 0.653; weight 7.5 keeps the handover income-neutral
        # (5.0 * 0.997 = 4.99 out, 7.5 * 0.653 = 4.90 in).
        # 2026-09-06 (Loop 22): target 0.38 -> 0.30 m, weight 7.5 -> 4.7, same reasoning as
        # jump_takeoff_apex above. 0.30 m is the trunk rise this robot reaches at the
        # actuator limit; 4.7 * 0.984 = 4.62 against the previous 7.5 * 0.618 = 4.64.
        # === 2026-09-15 報酬の全面棚卸し: weight 4.7 -> 2.5, target 0.30 -> 0.45 ===
        # 2つの問題が同時に起きていた。
        # (a) 飽和: Isaac の unaided_max_height_gain は 0.287 で、target 0.30 に対し
        #     Gaussian は 0.99 = ほぼ満点・勾配ゼロ。jump_takeoff_speed(Loop 11) と
        #     jump_takeoff_apex(Loop 16) を殺したのと同じ「飽和死」に既に入っていた。
        # (b) 天井: この Gaussian はピーク型なので 0.30 m を**超えると減点に転じる**。
        #     等エネルギーで 2*v_x*v_z/g が最大になるのは v_x = v_z で、現在の踏切
        #     エネルギー(mujoco 実測 v=(2.97,2.74))ではそれは apex ≈ 0.41 m に相当する。
        #     つまり目的関数が要求する高さより低い位置に、高さ報酬が蓋をしていた。
        # target 0.45 は測定済みの範囲(Isaac 0.287)の外側なので蓋が外れ、Gaussian は
        # 0.26 に戻って勾配が復活する。weight 2.5 は飛距離ベスト model_9400(Loop 10)の値。
        # 項自体は残す: この項だけ real_jump ゲートの外side で払われるので、跳躍が
        # ゲート未満に落ちたときの復帰路になっている(目的関数側はゲートで崖になる)。
        weight=2.5,
        params={
            "command_name": "jump_command",
            "target_height": 0.45,
            # 2026-09-15 スモークで実測して修正: height_scale 0.02 -> 0.06。
            # 0.02 は幅0.14m程度の**狭い帯**で、target を 0.45 に動かした結果
            # 実測 0.122 m の位置では exp(-(0.328^2)/0.02) = 0.005 = 実質ゼロになり、
            # 「real_jump ゲート未満からの復帰路」という残した理由が消えていた
            # (60 iter スモークの収入 0.0086)。0.06 なら 0.122 m で 0.17、
            # 0.287 m で 0.64 と、跳躍の全域で勾配が生きる。天井(ピーク0.45)は
            # 測定範囲の外のままなので、v_z への蓋にはならない。
            "height_scale": 0.06,
            "payout_window_s": 0.6,
        },
    )
    # --- landing quality (2026-09-03, v5) ---------------------------------------------
    # Both terms exist because of the same mujoco observation: the v4 policy arrived
    # trunk-first. Nothing in the v4 set shaped the landing at all -- the trunk hitting
    # the ground was covered only by the terminal ``base_contact`` check, which offers
    # no gradient for how to come down. See プロジェクト_StageB着地失敗の原因分析.md.
    #
    # Weights are sized against the ~1.15/s the policy earns by simply running through
    # the jump window: -30 makes a 4 cm trunk shortfall cost 0.048/s and a 10 cm one
    # 0.30/s, and +1.5 makes a clean feet-only touchdown worth more than a third of the
    # running income for as long as it is held.
    jump_trunk_clearance = RewTerm(
        func=mdp.jump_trunk_clearance,
        weight=-30.0,
        params={"command_name": "jump_command", "max_drop_m": 0.08, "max_shortfall_m": 0.15},
    )
    # 2026-09-06 (Loop 18): weight 1.5 -> 4.0. Falls have crept back from 0.4% (Loop 15,
    # foot clearance 0.43 m) to ~1.9% (Loop 17, 0.57 m) as the jump grew, and this is the
    # only landing term with a non-trivial income (+0.050; jump_landing_gear is down to
    # -0.002 because it already closed the front/rear gap, and trunk_clearance is -0.002).
    # Against a height budget of 1.77 the whole landing side was under 3%.
    # === Go2 サイクル2（2026-09-15 02:05）: 着地の飴を 4.0 -> 20.0 ===
    #
    # 着地がこの系列の唯一の不足。据え置きベスト `2026-09-13_14-19-07/model_29200` は
    # **飛距離 1.929m（目標2.06mまであと−6%）なのに着地 12%**。一方で収支を見ると
    # `jump_takeoff_distance` が +1.439 を払うのに対し **着地の飴はこの項の +0.133 だけ**で、
    # ポリシーにとって「遠くへ跳ぶ」は「うまく降りる」の10倍の価値がある。降りる動機が薄い。
    #
    # 機構を指定する報酬（踏切の対称性=Loop30、着地のスタッガ、着地衝撃=Loop33）は
    # **3つとも実測で外している**ので、ここでは新しい機構を足さず、
    # **既に実測で効いている結果指標（足から降りたか）の値段だけを上げる**。
    # 5倍にすると着地の飴は +0.67 相当で、踏切の飴の半分弱。跳躍を殺さない範囲。
    #
    # サイクル1（joint_vel を jump 窓の外に出す）は飛距離が動かず着地だけ半減して失敗、撤回済み。
    # **サイクル2は失敗、2026-09-15 04:20 に撤回（20.0 -> 4.0）。** mujoco 40試行 @vx2.8:
    #   基準 `model_29200`          1.929m / 着地12% / 成立85%
    #   `29600`(+400 iter)  **成立32%** / 着地2%  / 1.942m
    #   `30400`(+1200)      **成立10%** / 着地0%  / 1.132m
    #   `32199`(最終)        成立90% / 着地18% / **0.478m**
    # **着地の飴を5倍にしたら跳躍そのものが壊れた。** 着地して初めて入る大きな飴は、
    # 「思い切って跳ぶ」の期待値を下げる（崩れる跳躍は飴を丸ごと失う）。同じ変更を
    # Anaguma に入れた run は健全なので、これは Go2 の動作点（助走5.0m/s・頂点0.386）
    # 固有の壊れ方。着地を直接買う4つ目の試みも失敗。
    jump_landing_feet_first = RewTerm(
        func=mdp.jump_landing_feet_first,
        weight=4.0,
        params={
            "command_name": "jump_command",
            # 2026-09-03 (v6): these MUST be passed here rather than left as default
            # arguments. ManagerBase._prepare_terms resolves SceneEntityCfg objects only
            # where it finds them, in term_cfg.params (manager_base.py:397). Left as
            # defaults they keep body_ids = slice(None) = all 17 bodies, which made
            # any_foot_down and any_body_down the same expression -- the term was
            # identically zero for all 2996 iterations of v5.
            "foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "body_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base", "Head_.*", ".*_hip", ".*_thigh"]
            ),
        },
    )
    # One flight per jump. See mdp.jump_repeat_liftoff -- weight sized so a surplus
    # bounce held for a full 2.2 s window costs ~2.2, about two seconds of running
    # income, while a brief stumble costs little.
    jump_repeat_liftoff = RewTerm(
        func=mdp.jump_repeat_liftoff,
        weight=-1.0,
        params={"command_name": "jump_command"},
    )
    # 2026-09-03 (v7): the other half of distance = 2*v_x*v_z/g. Measurement showed the
    # take-off keeps only 0.84 m/s of a 3.3 m/s approach, so the robot is nearly jumping
    # from a standstill and no amount of extra height will produce a long jump.
    #
    # Weight 1.5 against jump_height's 2.5: at the measured v_x = 0.84 this pays
    # 1.5 * 0.42 = 0.63/s inside the jump window, rising to 1.5/s at the 2.0 m/s target,
    # so improving the take-off is worth about 0.9/s -- comparable to, but not larger
    # than, the 1.15/s the policy earns by simply running through the window. Height
    # stays the larger term because a jump that is fast but flat fails real_jump and
    # earns nothing here at all.
    #
    # 2026-09-04 (v8): weight 1.5 -> 2.5, to match jump_height's marginal pull.
    #
    # v7 did move the take-off speed (0.84 -> 1.28 m/s) but it plateaued at ~1.25 well
    # short of the 2.0 target, while distance stayed flat at ~0.28 m. Differentiating
    # both terms at the v7 end state (v_x 1.28, v_z 1.43, h 0.090) explains why:
    #
    #   distance = 2*v_x*v_z/g  ->  d(dist)/dv_x = 0.291,  d(dist)/dv_z = 0.261 m per m/s
    #                               (the objective values the two factors almost equally)
    #   jump_height  (w 2.5, exp) ->  dR/dv_z = 1.139 per m/s
    #   jump_takeoff (w 1.5, lin) ->  dR/dv_x = 0.750 per m/s
    #
    # So the reward was paying 1.52x more for a m/s of v_z than for a m/s of v_x, even
    # though the objective is indifferent between them -- and v_x is the factor that is
    # further from its target (30% of the 4.3 m/s the 2.06 m goal assumes, vs 32% for
    # height). Equalising the two gradients gives 1.5 * 1.52 = 2.28; rounded to 2.5 so
    # the two factors of the product carry the same weight, as the physics does.
    #
    # This is the only change in v8: same shape (clipped-linear), same 2.0 target, same
    # real_jump gate, so the prediction is directly checkable -- v_x should resume
    # climbing past 1.28, and distance with it.
    # 2026-09-05 (Loop 11, v10): weight 2.5 -> 0.0, replaced by jump_takeoff_distance
    # below. Kept in the config at weight 0 for the same reason jump_distance is, and
    # because the reason it is being retired is worth keeping next to it: by the end of
    # Loop 10 the policy took off at v_x = 2.158 m/s against this term's 2.0 m/s target,
    # so clamp(v_x/2.0) was pinned at 1.0 -- a flat +2.5 with no gradient in either
    # direction. It did its job (v_x 0.84 -> 2.16 across Loops 8-10) and is now done.
    jump_takeoff_speed = RewTerm(
        func=mdp.jump_takeoff_speed,
        weight=0.0,
        params={"command_name": "jump_command", "target_speed": 2.0},
    )
    # 2026-09-05 (Loop 11, v10): the objective itself, dense at the take-off.
    #
    # Loop 10 fixed the approach (v_x 1.222 -> 2.158) and in doing so inverted the
    # imbalance it had inherited: v_z fell 1.624 -> 1.301, and at C = 7.27 the ballistic
    # distance is 0.572 m where an even split would give 0.741 m. So there is 30% in the
    # split again -- the mirror image of Loop 9, which measured +4% and correctly left it
    # alone. See mdp.jump_takeoff_distance for why the product is rewarded rather than
    # either factor, and why this was chosen over simply raising jump_height's weight.
    #
    # Weight 9.0 is sized to hand over from jump_takeoff_speed with the in-window income
    # unchanged at the Loop 10 end state: that term paid a saturated 2.5, and this one
    # pays 9.0 * (0.572 / 2.06) = 2.50. Unlike its predecessor it then keeps climbing --
    # at the 2.06 m goal it would pay 9.0, which is the point.
    # 2026-09-05 (Loop 12, v11): weight 9.0 -> 0.0, replaced by jump_takeoff_apex below.
    # It did move v_z (1.301 -> 1.378) and set the project distance record (0.735 m), so
    # it is not being retired for failing. It is being retired because the objective has
    # changed to height, and because paying for the product 2*v_x*v_z/g always buys the
    # cheaper factor first: at a 2.1 m/s approach that is v_x, and the split v_x/v_z came
    # out of Loops 9, 10 and 11 unchanged at 1.66-1.68 every time. Kept at weight 0 so
    # the metric keeps being logged and the distance route can be resumed with one edit.
    # 2026-09-13 (Loop 47): weight 0.0 -> 74.0. The objective goes back to distance.
    #
    # tanaka: "飛距離のみを追い求めて" -- height is explicitly not wanted any more, so the
    # swap Loop 12 made (distance -> apex) is undone. Loop 12's reason for leaving was
    # that paying for the product 2*v_x*v_z/g buys the cheaper factor first, and at a
    # 2.1 m/s approach that was v_x. That condition no longer holds: fourteen loops of
    # the height objective have pushed the split the other way, to v_x 1.665 against
    # v_z 2.623 (Isaac, Loop 46 model_27800). The cheap factor now IS v_x, which is the
    # one distance wants bought.
    #
    # Weight 74.0 hands over with the in-window income unchanged at Loop 46's end state:
    # the apex term paid 50 * 0.877 * 0.727 (progress * approach gate) = 31.9, and this
    # pays 74 * (0.8905 / 2.06) = 32.0. Day one is reward-neutral; what changes is the
    # direction that earns more from here. And the gradient is restored -- apex sat at
    # 0.877 of full (nearly clamped, Loop 36 noted the same), this sits at 0.432.
    #
    # No approach gate on this term, unlike the apex one. Distance already contains v_x,
    # so a standing jump scores zero by construction; a gate would charge for the same
    # thing twice.
    # 2026-09-13 (Loop 48): target 2.06 -> 2.75 m, weight 74 -> 99 (income-neutral).
    #
    # Loop 47 worked: mujoco distance 0.970 -> 1.929 m over 40 trials, from v_x 1.25 ->
    # 2.97 against v_z 3.14 -> 2.74. But the ballistic product it is paid on now reads
    # 2*2.97*2.74/9.81 = 1.66 m against the 2.06 target -- progress 0.806, i.e. the term
    # is running out of room the same way jump_takeoff_apex did at 0.877 (Loop 36).
    # A clamped term does not pull.
    #
    # 2.75 is chosen to be reachable, not aspirational. The energy method puts Go2's
    # ceiling at 3.89 m for a 5.3 m/s approach, and these policies are already running at
    # 5.00 m/s in mujoco, so 2.75 sits inside the machine's range -- unlike Loop 21's
    # apex target, which was set past the then-ceiling and walked v_z down from 2.70 to
    # 2.05 chasing it. At the current operating point the new progress is 0.604, with
    # room above it.
    #
    # Weight 99 keeps day-one income unchanged: 74 * (1.66/2.06) = 59.6 out,
    # 99 * (1.66/2.75) = 59.8 in.
    #
    # Not changed, deliberately: the split. v_x 2.97 against v_z 2.74 is close to the
    # even split that maximises the product at fixed take-off energy, so there is nothing
    # left to win by re-balancing -- what is left is total take-off energy.
    jump_takeoff_distance = RewTerm(
        func=mdp.jump_takeoff_distance,
        # 2026-09-13 (Loop 49): reverted to Loop 47's 2.06 / 74.0. Raising the ceiling to
        # 2.75 did not buy distance -- it bought a collapse. Loop 48 peaked at +2.5% over
        # its start and then fell apart: mujoco distance 1.929 -> 1.852, real-jump rate
        # 85% -> 72%, falls 0.003 -> 0.115 over the back half of the run.
        #
        # The mechanism, which is new and worth stating: **a product reward, once one
        # factor is at its hardware limit, will spend the other factor to keep pushing
        # the one that cannot move.** v_x is now capped -- these policies approach at
        # 5.24 m/s against Go2's measured 5.3 m/s maximum -- so the term's remaining
        # gradient went into the take-off posture, buying v_x +5% at the cost of v_z -6%,
        # and the product went DOWN. The mirror of Loop 12's "the product buys the cheaper
        # factor": there the cheap factor was the useful one, here it is exhausted.
        weight=74.0,
        # === Go2 サイクル4（2026-09-15 06:36）: 目標 2.06 -> 1.20 に下げて飽和させる ===
        # 着地を直接買う試みは5回とも失敗した（踏切対称性/スタッガ/着地衝撃/着地の飴5倍/v_z上限）。
        # 残るのは「距離を伸ばす動機のほうを切る」間接策。Isaac 実測 1.06〜1.29 に対し 1.20 は
        # 現在の動作点の内側〜すぐ上で、勾配はほぼ無くなる。飛距離 1.929m は条件 2.06m の 94% まで
        # 来ていて、足りないのは着地 12% → 30%。Anaguma サイクル4と同じ賭けを Go2 でも張る。
        # === 2026-09-15: サイクル4 は失敗、2.06 に戻す ===
        # 発散前の健全な窓で測っても 1.929m -> 1.244m(着地 12% -> 5%)。踏切エネルギーが
        # 単調に縮んだ(v_z 2.74 -> 1.93)。狙いどおり「1.20 で満点」にした結果、それ以上
        # 遠くへ跳ぶ動機が消えただけだった。
        # さらに重要: 目的関数を飽和させたことが**発散の原因**でもあった。PPO cfg の
        # entropy_coef のコメントにある崩壊機構 -- タスク報酬が平らになると生きている
        # 勾配がエントロピー項だけになり、方策が広がる方向にしか動けなくなって崩壊する --
        # の署名(action noise std 0.34 -> 0.79、action_rate 爆発)が出ている。
        # 発散した iteration 数はサイクル2=16 / 3=88 / 4=2416 で、飽和の度合いと一致する。
        params={"command_name": "jump_command", "target_distance": 2.06, "payout_window_s": 0.6},
    )
    # === 2026-09-15 サイクル7: 踏切エネルギー C = v_x^2 + v_z^2 を払う新項 ===
    #
    # このプロジェクトの3ループがかりの結論は「配分比は報酬では動かず、伸びるのは常に C」
    # なのに、**C を払う項が今まで無かった**。目的関数の積 2*v_x*v_z/g は固定 C の中で
    # v_x = v_z へ寄せる勾配しか持たない。実測でも距離は C で説明できる:
    #   C=14.2 -> 1.674m / C=12.0 -> 1.761m / C=10.1 -> 1.389m / C=8.9 -> 1.310m
    # weight はスモークで実測して決める（目標は収入 0.4〜0.6 = 目的関数 1.47 の 1/3 程度）。
    # ゲートと払い出し窓は jump_takeoff_distance と同一にしてある。
    jump_takeoff_energy = RewTerm(
        func=mdp.jump_takeoff_energy,
        # スモーク実測: weight 10.0 で収入 +0.3446（unaided_takeoff_energy 10.50）。
        # progress は clamp 済みなので収入は weight に線形（+0.0345/単位）。
        # 14.0 で +0.48 = 目的関数 1.74 の約 28%。狙いの 0.4〜0.6 の帯。
        weight=14.0,
        params={"command_name": "jump_command", "target_energy": 16.0, "payout_window_s": 0.6},
    )

    # 2026-09-05 (Loop 12, v11): the height objective, dense at the take-off.
    #
    # Weight 9.3 hands over from jump_takeoff_distance with the in-window income
    # unchanged at the Loop 11 end state: that term paid 9.0 * (0.735 / 2.06) = 3.21, and
    # this one pays 9.3 * (0.0968 / 0.28) = 3.22, where 0.0968 m is the ballistic apex of
    # the measured v_z = 1.378 m/s. So the swap is reward-neutral on day one and the only
    # thing that changes is which direction earns more from here.
    #
    # See mdp.jump_takeoff_apex for why this is not the same lever as raising
    # jump_height's weight (that Gaussian's pull weakens as the target approaches; this
    # one's grows with v_z), and why removing v_x from the objective is the point rather
    # than a side effect.
    # 2026-09-06 (Loop 16): target 0.28 -> 0.45 m, weight 9.3 -> 15.0.
    #
    # The term is about to die of its own success: Loop 15 ends at v_z = 2.301 m/s, a
    # ballistic apex of 0.2694 m against the 0.28 m target -- **0.962 of full**, where the
    # clamp is close enough that there is almost no gradient left. That is the same
    # failure that killed jump_takeoff_speed in Loop 11 (pinned at 1.0, paying a flat
    # constant), and this project's own rule is to suspect saturation before touching a
    # weight when a number stops moving.
    #
    # 0.45 m puts the term back at 0.599 of full with room above it. Weight 15.0 keeps the
    # handover income-neutral: 9.3 * 0.962 = 8.95 out, 15.0 * 0.599 = 8.99 in. The term
    # stays bounded at 1.0, so the worst case is 15.0/s rather than an unbounded pull.
    jump_takeoff_apex = RewTerm(
        func=mdp.jump_takeoff_apex,
        # 2026-09-06 (Loop 17): 15.0 -> 30.0. The payout is now bounded to 0.6 s after the
        # take-off (see mdp.jump_takeoff_apex), which measured at 43% of the previous
        # income on the same policy. 30.0 restores about 85% of it -- deliberately not
        # 100%, because Loop 16 showed this side of the budget was already large enough to
        # drown every landing term (its income 0.75 -> 1.93 while all four landing terms
        # sat below 0.03 in magnitude).
        # 2026-09-06 (Loop 22): target 0.45 -> 0.37 m, weight 30.0 -> 25.0. 0.37 m is the
        # ballistic apex of v_z = 2.70 m/s, which is what this robot actually reaches with
        # its knee actuators saturated at 45.43 N.m. Asking for 0.45 kept a live gradient
        # toward a speed the hardware cannot produce, and Loops 20 and 21 both wandered
        # off the ceiling rather than sitting on it (v_z 2.70 -> 2.05 in Loop 21, with the
        # whole landing budget at only -0.07 -- it was not the landing terms squeezing it).
        # Targeting the ceiling turns this from a term that pushes into a term that HOLDS:
        # saturated at the limit, with a restoring pull the moment v_z drops below it.
        # 25.0 keeps the income unchanged at the operating point (30 * 0.826 = 24.8).
        # 2026-09-10 (Loop 36): target 0.37 -> 0.40, weight 25 -> 27 (budget neutral at the
        # operating point). At v_z 2.71 the 0.37 target gives progress 1.01 -- SATURATED,
        # so the term has no upward pull and every run resumed from model_27200 has let the
        # height fall while distance rose. Loop 22 saturated it on purpose ('holding, not
        # pushing') but a clamped term does not hold. 0.40 = v_z 2.80, just above the
        # measured 2.71 and inside reach: Loop 32 found the high posture at 44.7 N.m against
        # the 45.43 knee limit, i.e. headroom. Deliberately NOT Loop 21's 0.45, which was
        # far past the then-ceiling and walked v_z down to 2.05.
        # 2026-09-10 (Loop 37): 27 -> 50. Four runs resumed from model_27200 have all let
        # the height decay from 0.29 to ~0.23 regardless of what was changed (landing
        # impact, knee contact, pinned approach, apex target). The decay is not caused by
        # any of those -- model_27200's 0.291 was a transient peak inside a 500-iteration
        # probe, and ~0.23 is where this reward configuration's optimum actually sits.
        # Moving the optimum needs the height BUDGET raised, not the target moved: at 27
        # this term pays +1.2 per episode against track_lin_vel_xy's +1.05, and the speed
        # term pays on every step while this one pays only inside a jump window.
        # 2026-09-13 (Loop 49): 0.0 -> 40.0, target 0.40 -> 0.50, approach gate disabled.
        #
        # This is NOT a return to the height objective. The objective is still distance;
        # this term is back because distance = 2*v_x*v_z/g and **v_x has run out**. These
        # policies approach at 5.00-5.24 m/s against Go2's measured 5.3 m/s ceiling, so
        # the only factor with room left is v_z -- and v_z has fallen every loop since the
        # objective changed: 3.14 (Loop 46) -> 2.74 (Loop 47) -> 2.58 (Loop 48). Holding
        # v_x at 2.97 while putting v_z back to 3.14 is worth 1.66 -> 1.90 m of ballistic
        # product, more than any other lever in the set can offer.
        #
        # target 0.50 m is the apex of v_z = 3.13 -- what this robot demonstrably did in
        # Loop 46, a level it has reached rather than an aspiration (contrast Loop 21, and
        # Loop 48 immediately above). At the current v_z = 2.58 it reads 0.674, a live
        # gradient rather than the clamp that stalled both objectives before.
        #
        # Weight 40 against distance's 74: at the operating point distance pays
        # 74*(1.64/2.06) = 58.9 and this pays 40*0.674 = 27.0, so distance stays the larger
        # term by ~2:1 and this reads as a constraint on HOW the distance is earned rather
        # than as a competing objective.
        # 2026-09-14: 40.0 -> 0.0, for the from-scratch Stage B run. Loop 49's reasoning
        # above was sound on the ballistic arithmetic and wrong in mujoco. Measured over
        # 40 trials each, same seeds, same day:
        #     Loop 47 (this term 0)  1.929 m, jumps 85%, LANDS 12%
        #     Loop 49 (this term 40) 1.991 m, jumps 70%, LANDS  0/40
        # 3% more ballistic product bought with every landing in the sample. v_z is not a
        # free factor: it buys flight time, and flight time is what the landing has to
        # survive. Distance-only is the configuration that produced the best machine
        # result this project has, so it is the one the clean run starts from.
        weight=0.0,
        params={
            # Loop 31: pay the take-off in proportion to the approach speed it was
            # asked for at. See mdp.jump_takeoff_apex -- nothing in the set had ever
            # required the jump to happen while running, and a standing jump is easier.
            # 2026-09-13 (Loop 46): 2.0/2.5 -> 2.6/3.1. Distance is back as the objective
            # and 2*v_x*v_z/g is linear in v_x, which is the only factor still moving:
            # Loop 31 raised the approach 2.26 -> 2.56 and distance went 0.701 -> 0.773,
            # then Loop 32 reached 2.78 and 0.814 -- three measurements, no sign of a
            # knee. v_z by contrast has sat at 2.68-2.75 for fourteen loops.
            # The gate must sit above what the robot already does or it is a dead term
            # (Loops 11/16/18). At the measured 2.78 approach the new band pays 0.36 of
            # full, so there is headroom without the payout collapsing.
            # Deliberately NOT Loop 44's move (dropping track_lin_vel_xy's weight): that
            # released the ceiling to 4.4 m/s and those individuals landed in mujoco only
            # 24-36% of the time. This raises the ASK while the tracking term still holds
            # the approach to its command.
            # 2026-09-13 (Loop 49): gate switched off (lo 0.0 / hi 0.1 => always 1).
            # The gate exists to stop a standing jump being paid like a running one, and
            # jump_takeoff_distance already refuses to pay a standing jump by construction
            # -- it contains v_x. Left on it would charge the approach twice and, at a
            # 5 m/s approach, be saturated anyway.
            "approach_lo": 0.0,
            "approach_hi": 0.1,
            "command_name": "jump_command", "target_apex": 0.50, "payout_window_s": 0.6},
    )
    # Sparse: the task objective itself, paid at a scoring landing. Weight 20 makes a
    # 1.0 m jump worth 0.4 reward and a 2.0 m jump 0.8 -- deliberately modest, because
    # a large sparse weight is what made the previous version farmable by falling over.
    # 2026-09-03 (v6): weight 20 -> 0. One objective at a time, per the 09-03 review.
    # With the approach speed now FIXED (see RobotEnvCfgLongJump.__post_init__), distance
    # = 2*v_x*v_z/g has only one free variable left, and it is v_z = sqrt(2*g*h). So
    # optimising height at a fixed speed IS optimising distance, and keeping a second
    # term for it only adds a way for the two to disagree. It stays in the config at
    # weight 0 so the metric keeps being logged.
    jump_distance = RewTerm(
        func=mdp.jump_distance_reward,
        weight=0.0,
        params={"command_name": "jump_command", "max_distance": 3.0},
    )
    # 2026-09-06 (Loop 13): stop paying the policy to hold its forward speed at the exact
    # moment the task needs that speed converted into vertical speed. Same treatment
    # lin_vel_z_l2 got in v4, applied to the other side of the same trade -- see
    # mdp.track_lin_vel_xy_exp_grounded.
    #
    # Measured, so the claim is sized honestly: gating removes only 2.5% of the term's
    # episode income (0.431 -> 0.420), NOT the ~10% the jumping_fraction would suggest --
    # because a robot converting forward speed into vertical speed is already tracking
    # badly, so exp(-error/std^2) was already small there. The point is the GRADIENT, not
    # the level: while the term was live, every m/s given up in the take-off was pulled
    # back by dR/de = -4R, and that pull is what Loop 12 had to fight to move v_z at all.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp_grounded,
        # 2026-09-10 (Loop 39): 1.5 -> 0.8. Six runs have all decayed the height back to
        # ~0.23 whatever was changed, and doubling the apex budget did not move the optimum.
        # So reduce what competes instead of adding more. This term pays +0.8-1.0 on EVERY
        # step; the jump terms only pay inside a window and are diluted ~25x in episode
        # terms. Tonight's objective is height and distance is secondary, so the speed
        # pressure is the thing to give up. Watch the approach: if it falls under ~2.2 the
        # jump loses the run-up that Loop 32 showed it needs.
        # 2026-09-10 (Loop 41): 0.8 -> 1.2. At 0.8 the tracking term is loose enough that the
        # approach overshoots the U(2.8, 3.3) command all the way to 4.4 m/s, and tonight's
        # three data points say that is past the height optimum: approach 2.5 -> height
        # 0.27, 3.5 -> 0.33, 4.4 -> 0.30. 1.2 aims to hold it near 3.5. Not back to 1.5,
        # which squeezed the approach to 2.5 and was the cap that stalled six loops.
        # 2026-09-10 (Loop 45): back to 1.5. Lowering this to 0.8 removed the cap on the
        # approach and let it run to 4.4 m/s, which looked like a breakthrough in Isaac
        # (height 0.338, distance 1.010, both records) and was the opposite of what tanaka
        # asked for -- the instruction was to PIN the approach at its best value. mujoco
        # settled it: the 4.4 m/s individuals land 24-36% and jump 0.12 m, the 2.2 m/s ones
        # land 36-42% and jump 0.22 m. The Isaac gains were not real.
        weight=1.5,
        params={"command_name": "base_velocity", "jump_command_name": "jump_command", "std": math.sqrt(0.25)},
    )
    # 2026-09-06 (Loop 13): every real jump lands rear-first (measured 1.00) at about 10
    # degrees nose-up (projected gravity x = -0.173), which is tanaka's "the rear legs
    # touch down at once and the front legs get slammed in". See mdp.jump_flight_pitch.
    #
    # Weight -4.0: at the measured -0.173 the term reads (0.173/0.35)^2 = 0.244, so it
    # charges 0.98 per second in the window against the ~6.0/s jump_takeoff_apex pays
    # there -- about 16%, enough to shape the take-off without competing with the height
    # objective it is meant to support.
    jump_flight_pitch = RewTerm(
        func=mdp.jump_flight_pitch,
        weight=-4.0,
        params={"command_name": "jump_command", "max_tilt": 0.35},
    )
    # 2026-09-06 (Loop 14): the landing order, which Loop 13 proved is a leg problem
    # rather than an attitude problem -- attitude was fixed and rear_first stayed at 1.00.
    # See mdp.jump_landing_gear. Weight -3.0, sized in the smoke against the ~6/s the
    # height term pays in the window; only charged while descending, so the tuck that buys
    # the foot clearance on the way up is untouched.
    jump_landing_gear = RewTerm(
        func=mdp.jump_landing_gear,
        weight=-3.0,
        # max_delta 0.35, NOT the 0.12 of leg travel it was first sized at. Measured, the
        # front feet are 0.279 m above the rear ones at touchdown -- so at 0.12 the term
        # clips at 1.0 and pays a flat penalty with ZERO gradient, which is precisely the
        # dead-term failure Loop 11 diagnosed in jump_takeoff_speed. 0.35 keeps the
        # current 0.279 at 0.635 of full, still climbing in both directions.
        # 2026-09-06 (Loop 20): two-sided with a 0.05 m deadband, and max_delta 0.35 ->
        # 0.15 so the far side is actually charged. Measured after Loop 19: the gap is
        # -0.10 to -0.15 m (front feet BELOW the rear, nose-down) and falls rose 0.84% ->
        # 2.2%. At -0.125 the new form reads ((0.125-0.05)/0.15)^2 = 0.25, i.e. -0.75/s.
        # 2026-09-06 (Loop 21): deadband 0.05 -> 0.06 and the band is now asymmetric
        # ([-0.06, 0] m), so it no longer disagrees with jump_landing_order over the
        # 0..+0.05 m region where the front feet are above the rear ones.
        params={"command_name": "jump_command", "max_delta": 0.15, "deadband": 0.06},
    )
    # 2026-09-06 (Loop 19): the landing ORDER, now that the geometry is fixed. See
    # mdp.jump_landing_order. Weight sized in the smoke against jump_landing_feet_first.
    jump_landing_order = RewTerm(
        func=mdp.jump_landing_order,
        weight=-4.0,
        params={"command_name": "jump_command"},
    )
    # 2026-09-09 (Loop 33). The landing arrives as one slam: measured gap between the first
    # front contact and the first rear contact is -6.9 ms (rear first, inside one 20 ms
    # control step), with 870 N peak summed foot force = 5.9x body weight. jump_landing_order
    # and jump_landing_gear have already fixed the ORDER and the HEIGHT -- the feet are in
    # the right places and still land together. Nothing in the set has ever seen the time
    # gap. Weight set from the smoke, not from a formula. See mdp.jump_landing_stagger.
    # DISABLED 2026-09-09 before ever running: the mechanism is wrong on this robot.
    # Per-env over ~700 landings the front/rear gap correlates with the impact at +0.20,
    # and front-first landings hit at 875 N against 792 N for rear-first -- staggering
    # makes it HARDER here. Kept at weight 0 with its measurement metrics live, as the
    # record of a design that was checked before it was run. See mdp.jump_landing_impact.
    jump_landing_stagger = RewTerm(
        func=mdp.jump_landing_stagger,
        # 2026-09-09, from the first smoke: at 6.0 this paid +0.036, i.e. 2.7% of
        # jump_takeoff_apex (+1.349) and a quarter of jump_landing_feet_first (+0.144).
        # The landing lag moved -6.9 -> +1.3 ms and was then pushed back to -10.1 ms, with
        # the term's own income falling as it lost -- the reward pulled the right way and
        # was outvoted. That is the same 4%-of-the-take-off budget Loop 21 measured for the
        # whole landing side, and the reason no landing term in this project has ever
        # changed behaviour. 30.0 puts it at about +0.18, above jump_landing_feet_first and
        # roughly half of jump_foot_clearance -- the largest landing term rather than the
        # smallest. Loop 17 made a 15 -> 30 move of the same size for the same reason.
        weight=0.0,
        params={"command_name": "jump_command", "target_lag_s": 0.030,
                "tolerance_s": 0.030, "payout_window_s": 0.3},
    )
    # 2026-09-06 (Loop 26): the height a person actually sees. See mdp.jump_foot_clearance
    # -- the only height quantity left that the actuators do not cap. Weight sized in the
    # smoke against jump_height.
    jump_foot_clearance = RewTerm(
        func=mdp.jump_foot_clearance,
        # 2026-09-06 (Loop 28): target 0.65 -> 0.80 m, weight 6.0 -> 7.4. Loop 27 reached
        # 0.627 m, i.e. 0.965 of the 0.65 m target -- the same saturation that killed
        # jump_takeoff_speed (Loop 11) and jump_height (Loop 18). Unlike jump_takeoff_apex,
        # this quantity is NOT capped by the actuators, so the target should keep moving
        # rather than be pinned to a ceiling. Weight 7.4 keeps the handover income-neutral
        # (6.0 * 0.965 = 5.79 out, 7.4 * 0.784 = 5.80 in).
        # 2026-09-10 (Loop 43): 7.4 -> 16.0. Trunk rise has plateaued at 0.27-0.32 across
        # five runs, but foot clearance sits at 0.62 against a 0.80 target -- gradient still
        # alive, and Loop 26 identified it as the only height quantity the actuators do not
        # cap. If trunk rise is done, the visible height has to come from here.
        # 2026-09-10 (Loop 44): 16.0 was too much -- falls went 0.12% -> 22.9% and the run
        # collapsed at iteration 29250. 7.4 moved nothing. 10.0 is the midpoint.
        # === 2026-09-15 報酬の全面棚卸し: 10.0 -> 0.0 ===
        # この項は目的関数 2*v_x*v_z/g に**一切入らない**。導入理由(Loop 26)は
        # 「人が見て高く見えるのは足の下の隙間だから」＝高さ路線の見栄えの項で、
        # 飛距離路線に切り替えた Loop 47 以降も外し忘れていた。
        # 収入は +0.411/+0.433 と目的関数(+2.06/+4.81)の約20%、正の項では2番目の大きさ。
        # 有害な理由: 脚は Go2 の pitch 慣性の 79% を占める(リファレンス_空中姿勢制御)ので、
        # 空中で畳む動作は胴体を回す。mujoco の失敗は傾き 150〜176deg の前転が大半。
        # 決定打は Loop 44 の実測 -- この重みを 16.0 にしたとき転倒 0.12% -> 22.9% で run が崩壊した。
        # 飛距離に寄与せず、着地を壊し、収入だけ大きい。飛距離路線では不要。
        weight=0.0,
        params={"command_name": "jump_command", "target_clearance": 0.80, "payout_window_s": 0.6},
    )
    # Still 0, but for a measured reason now rather than a stale one. Loop 13 left it off
    # because "the measured peak is only 1.82 rad/s"; ``unaided_peak_pitch_rate`` (sampled
    # only while airborne) has since read 4.8-6.5 rad/s in every recent loop, so that
    # number was 2.8x out of date and the decision looked ripe for reversal.
    #
    # 2026-09-06 (Loop 29 smoke, 1024 envs from 12-00-05/model_26400): turned on at weight
    # -1.0 to size it, and the measurement says do not. Episode income was -0.0132 against
    # jump_takeoff_apex's +1.7278, i.e. 0.76%. Backing the raw term out of that: the jump
    # window is 0.104 * 20 s = 2.08 s per episode, so mean (w_y/6)^2 = 0.00635 and RMS
    # |w_y| = 0.48 rad/s. Reconciling that RMS with the 4.92 rad/s peak puts the peak at
    # ~0.02 s of the 2.08 s window -- ONE control step. The peak is a touchdown impulse,
    # not a flight rotation, so this term would spend its weight charging the landing
    # impact and would do nothing about attitude in the air.
    #
    # Which also settles what tanaka is seeing in mujoco ("it starts rotating in the air",
    # tilt 1.50-2.95 rad by 1.02 s): Isaac is NOT doing that, so it is a simulator
    # divergence rather than a hole in the reward set. See the actuator torque-speed note
    # in longjump_base_env_cfg.py -- Isaac derates torque above X1 = 13.5 rad/s and reaches
    # zero at 30, while go2.xml's ctrlrange is flat in speed.
    # 2026-09-09 (Loop 33). The landing is 846 N of peak ground reaction, 5.8x body weight,
    # while success_rate reads 97% -- it has never had a term for how hard the robot
    # arrives. Charges the objective and leaves the means to the policy, after two
    # externally-specified mechanisms (Loop 30 symmetry, landing stagger) both measured
    # wrong. Weight set from the smoke. See mdp.jump_landing_impact.
    # 2026-09-09 (Loop 34). Knees/elbows touch down on 99.8% of jumps; the existing
    # always-on undesired_contacts pays -0.040 for it, 3% of the take-off term. Weight set
    # from the smoke. See mdp.jump_leg_contact.
    jump_leg_contact = RewTerm(
        func=mdp.jump_leg_contact,
        # Smoke at -3.0 paid -0.042. The term is a per-step 0/1 flag, so income is exactly
        # linear in the weight: -0.014 per unit. -25.0 puts it at about -0.35, the band
        # where jump_foot_clearance (+0.39) sits and where a landing term has a chance of
        # being heard. (The stated rule was "3x and re-smoke", but 3x arithmetically lands
        # at -0.13, still under the target -- extrapolating to the intended band instead of
        # spending a smoke on a value already known to be too small.)
        # 2026-09-09: -25.0 moved the knee rate 0.986 -> 0.896 over 700 iterations and
        # paid for it out of the jump -- trunk rise -16%, clearance -12%. Third landing
        # penalty in a row whose cheapest response is to jump less. Kept at -5.0 so the
        # behaviour is priced (5x the old always-on term) without buying height down.
        weight=-5.0,
        params={"command_name": "jump_command"},
    )
    jump_landing_impact = RewTerm(
        func=mdp.jump_landing_impact,
        # Weight from the smoke, and the first estimate was out by 25x. The term reads 0.37
        # at the 846 N operating point, so -1.0 was expected to pay -0.37; it paid -0.015.
        # Episode_Reward averages over the whole episode, and a term that only pays for
        # 0.3 s after each touchdown is diluted by roughly the ratio of that window to the
        # episode. jump_landing_order sits at -0.0012 on weight -4.0 for exactly the same
        # reason -- the existing terms already showed this and the estimate ignored it.
        # That is 失敗パターン④ (design the payout PERIOD, not just the quantity) applied
        # to the weight rather than to the reward.
        #
        # DISABLED 2026-09-09 after 946 iterations at -25.0. The impact did not move at
        # all -- 836 N at iter 27300, 833 N at iter 28145, wandering inside the same band
        # the whole way -- while the trunk rise fell 0.260 -> 0.231 (-11%), lift-off v_z
        # 2.628 -> 2.503, and falls went 0.00% -> 2.4%. The reward could not reduce the
        # impact and paid for it out of the jump instead. Third mechanism tried against
        # this landing and third to miss.
        weight=0.0,
        params={"command_name": "jump_command", "max_impact_n": 1400.0, "payout_window_s": 0.3},
    )
    # === 2026-09-15 サイクル6: 0.0 -> -4.0、max_rate 6.0 -> 3.0 で有効化 ===
    #
    # サイクル5 の実測が根拠。同じ run の `model_12000`(mujoco 着地28%) と
    # `model_12200`(着地2%、飛距離は過去最高 1.761m) を Isaac の着地系メトリクス全部で
    # 比べても差が出ない -- upright_at_close 0.898/0.906、landing_max_tilt 0.248/0.252、
    # knee_contact 0.880/0.917、touchdown_vz -0.52/-0.60、absorb_ms 666/659。
    # **唯一はっきり差が出たのが回転レート**: peak_pitch_rate 5.50 -> 6.23、
    # flight_roll_rate 1.09 -> 1.23。着地が壊れた個体は空中でより速く回っていた。
    #
    # これは「着地の結果」を買う5連敗とは別の狙いになる。空中の角運動量は踏切で決まり、
    # 離地後は何をしても消せない(脚は pitch 慣性の79%)。だから踏切で回転を作らせない。
    # 関数側のコメントも「この項と高さの目的関数は同じ方向に引く」と書いている。
    #
    # max_rate 6.0 -> 3.0: 実測の flight_pitch_rate は平均 0.6、peak が 5.5〜6.2 なので
    # 6.0 のままだとピークで飽和(clamp 1.0)して 5.5 と 6.2 を区別できない。3.0 なら
    # 飛行中の常用域で二次的に効く。weight は 60 iter スモークで収入を実測して決める
    # (目標は -0.2〜-0.4 の帯。`jump_leg_contact` のコメントにある
    #  「着地系の項が聞こえる band」)。
    jump_pitch_rate = RewTerm(
        func=mdp.jump_pitch_rate,
        # === 2026-09-15 サイクル6の結果: -12.0 は失敗、0.0 に撤回 ===
        # 予測は「回転が下がれば同じ踏切エネルギーが並進に回り飛距離は 1.761m 以上」。
        # 実測は **飛距離 1.310m（-26%）**。peak_pitch_rate は 6.23 -> 4.2〜5.2 と
        # 狙いどおり下がったが、**離陸速度が両方縮んだ**（v=(2.76,2.09) -> (2.55,1.56)、
        # C = 12.0 -> 8.9）。事前に書いた反証条件「回転が下がったのに飛距離が縮んだら、
        # 回転は飛距離の副産物であり削ると踏切そのものが弱くなる」に該当したので撤回。
        #
        # **ただし着地には効いた**: 着地率 28% -> **68〜78%**（`model_12500` で78%、
        # Go2 のプロジェクト記録。従来最良28%）。跳躍成立も88〜98%で健全。
        # **着地を目的にするならこの項が唯一効いたレバー**なので、飛距離路線を降りる
        # 判断をするときは最初にここへ戻る。1500 iter 通して崩壊も起きなかった。
        weight=0.0,
        params={"command_name": "jump_command", "max_rate": 3.0},
    )
    # Loop 30 (2026-09-09). The one quantity that separates mujoco's landings from its
    # failures and is measurably present in Isaac (1.350 rad/s) while nothing charges it.
    # max_rate 2.2 keeps the operating point at 0.38 of the term instead of saturated --
    # see mdp.jump_roll_rate for why the take-off symmetry design was dropped in favour
    # of this one.
    jump_roll_rate = RewTerm(
        func=mdp.jump_roll_rate,
        weight=-4.0,
        params={"command_name": "jump_command", "max_rate": 2.2},
    )
    # Vertical-velocity penalty, now disabled while a jump is commanded.
    base_linear_velocity = RewTerm(
        func=mdp.lin_vel_z_l2_grounded, weight=-2.0, params={"command_name": "jump_command"}
    )
    # Same params-resolution fix as jump_landing_feet_first: left as a default argument
    # this had been computing std over all 17 bodies instead of the 4 feet since v2.
    # === Go2 サイクル1（2026-09-15）: joint_vel の課金を jump 窓の外に出す ===
    #
    # `2026-09-14_17-02-52/model_10899`(mujoco 1.354m/着地28%/成立100%) は Isaac 上で
    # 飛距離 0.799 m・iter 10400 以降ほぼ不動で頭打ちになった。そのときの収支で
    # **単独最大のペナルティが joint_vel の −0.2416**(次点 action_rate −0.1773)。
    # 踏切は関節を最速で伸ばす動作で膝は 37〜58 rad/s 回るので、**跳ぶほど課金される**。
    # lin_vel_z_l2_grounded(v4)・track_lin_vel_xy_exp_grounded(Loop 13) と同じ扱いにする。
    # Anaguma 側は影響を受けないよう素の joint_vel_l2 を明示的に固定してある。
    # **サイクル1は失敗、2026-09-15 02:05 に撤回。** mujoco 40試行 @vx2.8 で
    #   基準 `17-02-52/model_10899`  1.354m / 着地28% / 成立100% / 頂点0.267
    #   サイクル1 `23-54-38/model_13400` 1.354m / **着地15%** / 成立100% / 頂点0.303
    #   サイクル1 `23-54-38/model_13898` 1.338m / 着地10% / **成立52%**
    # **飛距離は1mmも動かず、着地だけが半減した。** Isaac 側は距離 0.799 -> 0.903 と
    # 伸びて見えたので、Isaac の指標が実機を予測しない例がまた1つ増えたことになる。
    # 踏切の正則化を外すと v_z が上がり(1.79 -> 1.91)、頂点が上がった分だけ着地が難しくなった。
    # joint_vel は素の定義に戻す。
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    # === Go2 サイクル3（2026-09-15 04:20）: 踏切 v_z の上限超過分を罰する ===
    #
    # 着地を直接買う試みは4回とも失敗した（踏切対称性/スタッガ/着地衝撃/着地の飴5倍）。
    # 5回目は着地ではなく**着地を難しくしている入力**を縛る。同日の mujoco 5個体で
    # 頂点と着地率は例外なく逆相関（0.243→10% / 0.267→28% / 0.303→15% / 0.362→2% / 0.386→12%、
    # 対して Anaguma の成功個体は 0.199→90%）。飛距離は積なので、v_z を削った分を v_x で
    # 買い直せば距離を保ったまま降りやすくなるはず、というのがこの項の賭け。
    # cap 2.0 は基準個体の離陸 v_z 2.74 に対して −27%。超過分だけ二乗で課金する。
    # weight −10 は「窓の長さ/エピソード長」で薄まる分を見込んだ見積もり（apex 項が
    # weight 50 で +1.68 払った実績から逆算して収入 −0.2 前後）。スモークで実測して確認する。
    # **サイクル3も失敗、2026-09-15 06:35 に撤回（−10 -> 0）。** 項は Isaac 側で狙いどおり
    # 発火し（収入 −0.08〜−0.11）、離陸の配分も v_x 2.77→3.16 / v_z 2.09→2.25 と横に寄ったが、
    # mujoco 実測は `29600` 1.376m/着地2%、`31200` 1.680m/着地8%、`32199` 1.106m/着地5% で
    # **着地はまったく改善せず飛距離だけ落ちた**（基準 1.929m/12%）。mujoco 側の頂点は
    # 0.25〜0.31 のままで、Isaac の v_z を削っても mujoco の頂点は動いていない。
    jump_liftoff_vz_cap = RewTerm(
        func=mdp.jump_liftoff_vz_cap,
        weight=0.0,
        params={"command_name": "jump_command", "v_z_cap": 2.0, "payout_window_s": 0.6},
    )
    jump_dense_reward = RewTerm(
        func=mdp.jump_dense_reward,
        weight=0.5,
        params={
            "command_name": "jump_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )


@configclass
class TerminationsCfgLongJump(TerminationsCfg):
    """Orientation termination relaxed only for a bounded window after the trigger.

    2026-09-02: ``relax_window_s`` added. The relaxation used to last as long as
    jump_command==1, and the old command could only clear once the robot was upright
    again -- so falling over bought permanent immunity from this termination. 1.0 s
    comfortably covers a real jump (0.12 s crouch + 0.22 s launch + flight) while
    ending the episode of anything still on its side afterwards.
    """

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation_grounded,
        params={"limit_angle": 0.8, "command_name": "jump_command", "relax_window_s": 1.0},
    )


@configclass
class CurriculumCfgLongJump(CurriculumCfg):
    """Command-range, jump-target and assist-decay curricula.

    ``jump_vel_target_levels``' gate was fixed on 2026-08-31 to normalise by lift-off
    attempts rather than episode duration (the sparse reward fires on at most a handful
    of single steps, so dividing by 20 s crushed the ratio ~3 orders of magnitude below
    any reachable threshold). It still never moved in the v2 run -- but that is now
    explained: with min_air_time_s at 0.05 s there were essentially no lift-offs to
    judge. Left in place, with the assist now making real lift-offs happen.

    ``jump_assist_decay`` is new: it withdraws the EFGCL assist as the policy starts
    landing jumps unaided (tak's assist_force_decay, ported).
    """

    # 2026-09-03: jump_vel_target_levels removed. It ratcheted the horizontal lift-off
    # speed target, a quantity that is now not rewarded at all (see
    # RewardsCfgLongJump), and it never advanced off 1.5 in any of the three runs it
    # shipped in. The height target is a fixed point by design -- tak measured that
    # ranging it collapses learning.
    # 2026-09-03 (v6): DISABLED. Measured across v2/v3/v4/v5, this term promoted on
    # essentially every interval, so the command ceiling rose at a near-constant
    # 0.10-0.23 m/s per 100 iterations regardless of how the policy was doing -- an
    # open-loop ramp wearing a curriculum's clothes. In all four runs every performance
    # metric peaked at relative iteration 200-600 and declined monotonically from there
    # as the ramp continued, and in v5 (the first run able to measure it) the unaided
    # jump height fell from 0.098 m at a 1.0 m/s ceiling to 0.034 m at 4.6 m/s. Holding
    # the speed fixed is the single change with the most evidence behind it.
    lin_vel_cmd_levels = None

    jump_dense_reward_decay = CurrTerm(mdp.jump_dense_reward_decay)
    # 2026-09-03 (v5): the v4 run ended with assist_scale still at 0.94 after 3000
    # iterations, so the policy was graded almost entirely on what an external force was
    # doing for it (mujoco, which has no assist, showed almost no jump). Three changes:
    # the gate is now the real-jump rate rather than the landing-quality rate that never
    # crossed threshold; decay_step is doubled; and force_zero_at_step is a hard ceiling
    # that reaches 0 regardless of performance. 30_000 simulation steps is ~1250 PPO
    # iterations at 24 steps/iteration, leaving the final ~1750 iterations fully unaided.
    jump_assist_decay = CurrTerm(
        mdp.jump_assist_decay,
        params={
            "command_name": "jump_command",
            "success_threshold": 0.60,
            "decay_step": 0.04,
            "minimum_episodes": 1024,
            # 2026-09-05 (Loop 12): 30_000 -> 19_200 steps = 800 iterations at 24
            # steps/iter, so a 2000-iteration run trains unaided for its last 1200.
            # Loop 9 is the counter-example to copy away from: 1100 of its 1500
            # iterations still had the assist fading and the unaided numbers declined
            # monotonically through the 400 that were left.
            # 2026-09-14: 19_200 -> 36_000 (800 -> 1500 iterations). 800 was sized for a
            # resumed policy that only needed the teacher out of the way. From iteration 0
            # the take-off has to be learned first, and a 3000-iteration run still leaves
            # its whole second half unaided at 1500.
            "force_zero_at_step": 36_000,
        },
    )


@configclass
class RobotEnvCfgLongJump(RobotEnvCfgLongJumpBase):
    """Stage B: jump-integrated env, built on the Stage A flat-ground base.

    Inherits Stage A's corrected actuator model and domain randomisation via
    RobotEnvCfgLongJumpBase.
    """

    commands: CommandsCfgLongJump = CommandsCfgLongJump()
    observations: ObservationsCfgLongJump = ObservationsCfgLongJump()
    rewards: RewardsCfgLongJump = RewardsCfgLongJump()
    terminations: TerminationsCfgLongJump = TerminationsCfgLongJump()
    curriculum: CurriculumCfgLongJump = CurriculumCfgLongJump()

    # Approach speed, fixed rather than ramped (2026-09-03, v6).
    #
    # 3.3 m/s is where the best policy produced so far actually lives: v5's model_4300
    # measured an unaided 0.069 m jump carrying 0.408 m of distance at a 3.28 m/s
    # ceiling, and it is the first policy that looked right in mujoco. Freezing the
    # ceiling there keeps that distance while leaving height as the only thing training
    # can still move. The alternative considered was the paper's 2 m/s approach, which
    # gives more height (0.086 m at 2.08) but throws away distance we already have.
    # 2026-09-13 (Loop 46): 2.8 -> 3.3, restoring the band U(2.8, 3.3).
    # Loop 45 pinned the approach at 2.8 because Loop 44 had released it to 4.4 and
    # those policies stopped landing in mujoco. Pinning was right for the height
    # objective; for distance it removes the only remaining variable. A band whose
    # ceiling is 3.3 is still inside the range that lands (model_27200 runs at 2.78
    # and lands 42%), unlike the 3.5-4.4 that Loop 44 produced.
    APPROACH_SPEED_MS: float = 3.3

    # 2026-09-05 (Loop 10): the lower bound, previously 0.0.
    #
    # "Approach speed fixed at 3.3 m/s" was never true: the range was U(0.0, 3.3), so
    # 3.3 was a ceiling and the mean commanded approach was 1.65 m/s. The Loop 10
    # instrumentation measured what the robot was actually doing at the instant a jump
    # was triggered, over the holdout envs whose jumps cleared the real_jump gate:
    #
    #   speed at trigger          1.11 - 1.17 m/s
    #   speed entering take-off   1.07 - 1.13 m/s
    #   horizontal speed at lift-off  1.15 - 1.21 m/s   (v8 logged 1.222 -- same number)
    #
    # Two conclusions, both of which overturn the Loop 9 post-mortem's premise:
    #
    # 1. Horizontal speed is *preserved* through the take-off (1.11 -> 1.21, and the
    #    legs add v_z = 1.57 on top). There is no 62% braking loss to recover -- that
    #    figure came from comparing the take-off energy against a 3.3 m/s approach that
    #    was not happening. Loop 8's founding measurement ("only 0.84 m/s of a 3.3 m/s
    #    approach survives") was the same arithmetic on the same wrong premise.
    # 2. What is actually missing is the run-up. distance = 2*v_x*v_z/g is linear in
    #    v_x, and the robot has been jumping from ~1.1 m/s -- i.e. training a standing
    #    jump under a task named for a running one.
    #
    # So the range is narrowed to a band at the top rather than reaching down to zero.
    # Episodes still start from rest, so the acceleration transient is still trained;
    # what disappears is holding a slow cruise, which this task never wanted. This is the
    # seventh instance of this project's recurring failure -- a name ("助走 3.3 m/s")
    # that did not match what was being measured -- and the first one in the task
    # configuration rather than in a reward term.
    APPROACH_SPEED_MIN_MS: float = 2.8

    def __post_init__(self):
        super().__post_init__()
        # 2026-09-16: 帯を環境変数で上書きできるようにした（既定値は据え置き）。
        #
        # 記録個体 model_12700 の助走スイープ（mujoco 40試行 x コマンド 2.0-4.0、
        # eval_go2_approach_sweep/）で分かったこと:
        #   * コマンドは効く。d実測/dコマンド = 0.96、ただし実測は常に +0.4 m/s 上振れする。
        #   * 飛距離の山は実測 3.37 m/s ＝ この帯の上限 3.3 にぴったり乗る。
        #   * 山の先で崩れるのは助走ではなく踏切。離陸v_x/助走 の変換比が
        #     0.95 -> 0.85 -> 0.71 -> 0.46 と落ち、助走は 4.3 まで出るのに離陸は 1.99 まで下がる。
        # 山が学習分布の縁なのか物理限界なのかは、帯を上げて学習し直せば決着する。
        # そのための上書き口であって、既定値を変えるものではない。
        lo = float(os.environ.get("APPROACH_SPEED_MIN_MS", self.APPROACH_SPEED_MIN_MS))
        hi = float(os.environ.get("APPROACH_SPEED_MS", self.APPROACH_SPEED_MS))
        speed = (lo, hi)
        print(f"[longjump] approach band = {speed}", flush=True)
        self.commands.base_velocity.ranges.lin_vel_x = speed
        self.commands.base_velocity.limit_ranges.lin_vel_x = speed

        # 2026-09-16: 行動遅延の DR。**既定で入れる**（上の SIM_DT や助走帯と違い、
        # これは「既定値は据え置き」にしない）。
        #
        # 理由: 遅延ゼロは物理の理想化であって選択肢ではない。実機も mujoco の C++/DDS 経路も
        # 「観測した状態に対する指令が効くのは次の制御周期」で、構造的に最低1ステップ遅れる。
        # それまでのポリシーは遅延1ステップ(20 ms)で着地 48% -> 5%、飛距離 1.338 -> 0.359 m、
        # 飛距離最良個体は跳躍成立 90% -> 0% になり、C++ 経路では5回跳ばせて5回とも転倒した
        # (docs/go2_overnight_20260916.md)。学習にも評価にも遅延が無かったので、
        # **どちらを見てもこの脆さが見えなかった**のが根の問題。
        # 遅延なしで測りたいときだけ ACTION_DELAY_STEPS=0,0 で明示的に外すこと。
        # 2026-09-16: 着地を目的にする期間のための上書き口（既定値は据え置きの 0.0）。
        # この項は Loop 40 (2026-09-15) で **実測済み**: weight -12.0 / max_rate 3.0 で
        # 着地率 28% -> 68〜78%（プロジェクト記録）、代償は飛距離 -26%。
        # 「飛距離路線を降りる判断をするときは最初にここへ戻る」と当時書いた場所。
        w_pitch = os.environ.get("JUMP_PITCH_RATE_W")
        if w_pitch is not None:
            self.rewards.jump_pitch_rate.weight = float(w_pitch)
            print(f"[longjump] jump_pitch_rate weight = {self.rewards.jump_pitch_rate.weight}", flush=True)

        lo_d, hi_d = (int(x) for x in os.environ.get("ACTION_DELAY_STEPS", "0,2").split(","))
        cur = self.actions.JointPositionAction
        self.actions.JointPositionAction = mdp.DelayedJointPositionActionCfg(
            asset_name=cur.asset_name,
            joint_names=cur.joint_names,
            scale=cur.scale,
            use_default_offset=cur.use_default_offset,
            clip=cur.clip,
            delay_steps_range=(lo_d, hi_d),
        )


@configclass
class RobotPlayEnvCfgLongJump(RobotEnvCfgLongJump):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        # Mirror RobotPlayEnvCfg / RobotPlayEnvCfgLongJumpBase: start commands
        # at the trained ceiling instead of the curriculum's initial narrow
        # range (see フィードバック_resume時カリキュラムリセット.md).
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Show what the policy does unaided, not what the assist is still doing for it.
        self.commands.jump_command.initial_assist_scale = 0.0
