# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Injects ActorCriticEE/PPOEE into rsl_rl.runners.on_policy_runner's module
# namespace, where OnPolicyRunner resolves policy_cfg/alg_cfg["class_name"]
# via eval(). Must happen before any OnPolicyRunner is constructed; importing
# this module (which train.py/play.py do to resolve rsl_rl_cfg_entry_point)
# is early enough. See rsl_rl_ee.py's module docstring for the full design.
import rsl_rl.runners.on_policy_runner as _on_policy_runner_module  # noqa: E402
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ee import ActorCriticEE, PPOEE  # noqa: E402

_on_policy_runner_module.ActorCriticEE = ActorCriticEE
_on_policy_runner_module.PPOEE = PPOEE


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    # 2026-09-15 サイクル10: 100 -> 50。飛距離の最良個体は resume から +100〜200 iter の
    # 狭い窓に出る（サイクル5は+200、7は+100、9は+100で記録更新）ので、100刻みでは
    # ピークを踏み外す。収穫の解像度を上げるため 50 にする。
    save_interval = 50
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RslRlPpoActorCriticEECfg(RslRlPpoActorCriticCfg):
    """Adds the Explicit Estimator's hidden-layer sizing to the policy cfg."""

    class_name: str = "ActorCriticEE"
    estimator_hidden_dims: list[int] = MISSING


@configclass
class RslRlPpoAlgorithmEECfg(RslRlPpoAlgorithmCfg):
    """Adds the Explicit Estimator's own learning rate to the algorithm cfg."""

    class_name: str = "PPOEE"
    estimator_lr: float = MISSING


@configclass
class LongJumpPPORunnerCfg(BasePPORunnerCfg):
    """Stage B agent config: Explicit Estimator (ActorCriticEE/PPOEE) plus the
    extra ``estimator_target`` observation group's algorithm-set mapping.

    ``estimator_target`` is not one of rsl_rl's built-in ``default_sets``
    (only "critic" -- and "rnd_state" when RND is on -- get auto-resolved),
    so it must be listed here or it is silently never fed to anything.
    ActorCriticEE/PPOEE (see リファレンス_ExplicitEstimator実装仕様.md) read it via
    ``self.obs_groups["estimator_target"]``, mirroring how the stock class
    reads ``obs_groups["policy"]``/``obs_groups["critic"]``.
    """

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "estimator_target": ["estimator_target"],
    }

    policy = RslRlPpoActorCriticEECfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        # Phase 1 minimal label set sizing agreed 2026-08-27 (see
        # リファレンス_ExplicitEstimator実装仕様.md), same [256, 128] genesis_lr used.
        estimator_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmEECfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 2026-09-06 (Loop 24): 0.01 -> 0.002. The collapse signature that ended Loops 21,
        # 22 and 23 is ENTROPY RISING (21.0 -> 22.4) with estimator loss rising alongside
        # it, then the policy going. That is the opposite of convergence, and it is what an
        # entropy bonus does when it is the largest live gradient left.
        #
        # It became the largest live gradient by design: since Loop 22 both height terms
        # are targeted AT the machine ceiling (take-off torque pinned at the 45.43 N.m knee
        # limit for six loops), so they are deliberately flat there. Flat task reward plus a
        # 0.01 entropy bonus means the only thing still pushing is the one that widens the
        # action distribution -- and a policy balanced on a saturated actuator has nowhere
        # to widen into.
        #
        # This is the one mechanism the previous three interventions did not touch: Loop 21
        # changed the landing band, Loop 22 the height targets, Loop 23 the learning rate,
        # and all three collapsed identically at 700-1200 iterations after resume.
        # 2026-09-06 (Loop 25): 0.002 -> 0.005. At 0.002 the mechanism was confirmed --
        # entropy fell 20.7 -> 2.8 instead of climbing, the run held for 1900 iterations
        # against the 700-1200 of the previous seven, and falls went 0.77% -> **0.05%**.
        # But it then collapsed from the OTHER side: entropy bottomed at 2.8 (a nearly
        # deterministic policy), turned back up, and the policy went with it at iteration
        # 24900. Too little exploration is its own instability. 0.005 sits between the
        # runaway-up at 0.01 and the collapse-to-2.8 at 0.002.
        # 2026-09-06 (Loop 26): back to 0.002, the value that produced model_23600 (falls
        # 0.05%, v_z 2.731, success 97.4%). 0.005 held entropy in the target 8-15 band for
        # 1800 iterations but its best checkpoint was worse on every landing number
        # (falls 0.30%, success 96.8%). All three values collapse at 1800-1900 iterations
        # after resume, so the run length is capped below that rather than tuned further.
        # 2026-09-06 (Loop 27): 0.002 -> 0.005 again, but for a reason the earlier sweep
        # did not see: **entropy carries across resumes** -- it is the policy's own action
        # std, not optimiser state. Loop 26 started at entropy 6.7 (already decayed by
        # Loop 24's 0.002) and 0.002 drove it to 3.6, where the run collapsed after only
        # 1250 iterations instead of the usual 1800. Starting from a policy that is already
        # narrow, the low coefficient reaches the too-deterministic failure sooner.
        # 0.005 is the value that HELD entropy (11.8-12.0 for 1800 iterations in Loop 25),
        # which is what a policy entering at 3.6 needs.
        #
        # 2026-09-09 (Loop 30): back to 0.002. Same rule, different entry point -- the
        # rule is that the coefficient is chosen against the entropy the policy comes in
        # with, not in absolute terms. Loop 30 resumes from model_26600 at entropy 12.6,
        # and Loop 29 put the holding value (0.005) on exactly that policy and watched
        # entropy climb 12.5 -> 20.2 and collapse after 700 iterations. From 12.6 the
        # narrowing side is the correct choice, and it is also the value that produced the
        # project's best landing in Loop 24.
        #
        # 2026-09-09 (Loop 31): back to 0.005, same rule again. Loop 31 resumes from
        # model_27000, which Loop 30's 0.002 already narrowed (action noise std 0.72 ->
        # 0.38, entropy loss 4.7) and which then broke from the too-deterministic side at
        # iter 27950. A policy entering that narrow needs the holding value, not more
        # narrowing -- exactly the Loop 26 -> 27 situation.
        # 2026-09-10 (Loop 38): 0.005 -> 0.002. Five consecutive runs from model_27200 let the
        # height decay from ~0.29 to ~0.23 no matter what was changed (landing impact, knee
        # contact, pinned approach, apex target, apex weight), and all five share one
        # signature: entropy rising monotonically, 10.0 -> 11.6 and 10.6 -> 14.2. That is
        # the collapse signature Loops 17-27 identified, and it explains a decay that is
        # independent of the intervention. model_27200 enters at entropy ~10, so by the
        # rule established in Loop 26 the narrowing side is what it needs.
        # 2026-09-10 (Loop 42): 0.002 -> 0.005. Entropy has fallen to 2.6, the depth from
        # which Loop 24 reversed and collapsed, and the height has plateaued at 0.29-0.30
        # for two full runs. From an entry that narrow the rule says the holding side.
        # 2026-09-10 (Loop 43): back to 0.002. From entry 2.6 the 0.005 of Loop 42 expanded
        # rather than held -- entropy 2.6 -> 14.2 and the height fell to 0.267.
        # === 2026-09-15 サイクル11: 0.002 -> 0.0005 ===
        # 記録個体の収穫で分かったこと: **飛距離の平均を縛っているのは水準ではなく
        # ばらつき**。成功試行だけで見た飛距離の分布は
        #   記録 `model_12400`(std 0.31 時点): 平均1.825 / 標準偏差 **0.111** / 最大2.057
        #   サイクル10(std 0.43〜0.44 時点): 平均1.64〜1.75 / 標準偏差 **0.31〜0.40** / 最大 **2.93**
        # **単発では 2.9 m 跳べている**ので、2.06 m は物理的には届く。届かないのは一貫性。
        # そしてばらつきは方策エントロピーと一緒に増えている(std 0.31 -> 0.44)。
        # resume 直後の伸びはエントロピー上昇が作っているが、上がりすぎるとばらつきで平均を失う。
        # **狙いは std 0.30 前後の帯に長く留めること**。Loop 21〜27 は 0.002 と 0.005 の
        # 往復を7回やっているが、**0.002 より下は一度も試していない。**
        # === 2026-09-15 サイクル12: 0.0005 -> 0.002 に戻す ===
        # サイクル11の結果: エントロピー制御自体は成功した（方策 std が +250 iter まで
        # 0.31〜0.33 に留まり、0.002 のときの 0.43〜0.44 にならなかった）。
        # 飛距離のばらつきも下がった個体が出た（標準偏差 0.166 / 0.199）。
        # **だが平均も一緒に下がった**（1.580 / 1.532）。低ばらつき個体は低平均で、
        # 記録個体（平均1.825・標準偏差0.111）のような「両立」は出なかった。
        # → 伸びとばらつきは分離できない。記録を出した値に戻す。
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        # Placeholder, not yet reconciled against the conflicting numbers
        # noted in リファレンス_ExplicitEstimator実装仕様.md (genesis_lr's config
        # said 2e-4, its own docstring/README implied 1e-3 elsewhere).
        estimator_lr=1.0e-3,
        # 2026-09-06 (Loop 23): "adaptive" -> "fixed" at 1.0e-4.
        #
        # Every loop since 17 has run well for 700-1200 iterations and then collapsed, and
        # the learning rate is why. rsl_rl's adaptive schedule moves the LR by a factor of
        # 1.5 per update between 1e-5 and 1e-2, and in Loop 22 it read 1e-05 at iteration
        # 23000 and 3.8e-04 at 23300 -- a 38x excursion -- with the surrogate loss jumping
        # 15x at the same point and the collapse (v_x 1.48 -> 0.86, falls 2% -> 18%)
        # landing between them.
        #
        # This policy has no slack to absorb a step that size: the take-off has had its
        # knee actuators saturated at 45.43 N.m since Loop 18, so it sits on a boundary
        # rather than in a basin. 1e-4 is roughly the geometric middle of the range the
        # schedule actually used while the runs were healthy.
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
