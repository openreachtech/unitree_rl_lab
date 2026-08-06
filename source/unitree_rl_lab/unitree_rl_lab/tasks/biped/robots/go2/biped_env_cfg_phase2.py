"""Phase 2: forward/backward 2-leg walking, without leaning on a front foot.

Phase 1 (``biped_env_cfg.py``) achieves genuine 2-leg standing/walking in any
direction, but forward walking specifically kept touching a front foot down --
mechanically the easiest way to satisfy "don't fall" while still learning the
harder weight-shift a front-leg-free forward step requires. Reward-shaping
alone (reweighting ``front_contact_force``, command-distribution reshaping,
heading-command tracking, curriculum phases restricting the command range,
even hard terminations on excess lateral motion/heading error) never
eliminated it and twice caused catastrophic collapse when made strict enough
to matter (sandbox try12/try14 -- see ``sandbox/SUMMARY.md`` and the try*.py
history for the full experimental record).

This promotes the approach that finally worked (sandbox try13 -> try15 ->
try16): forward-only commands at a *slow* target speed, plus External Force
Guided Curriculum Learning (EFGCL) -- a decaying external force (forward +
up, see ``mdp.ForwardAssistVelocityCommand``) that physically assists the
weight-shift early in training, mirroring this codebase's own real-hardware-
deployed use of the same technique for backflip (``feat/jump`` branch,
``Unitree-Go2-Jump-Phase3``). Two things had to change together, not just
the assist force alone:

- The assist decays once a batch of episodes clears ``success_threshold``
  while keeping front-foot contact below ``front_contact_fraction_threshold``
  (see ``mdp.assist_force_decay``) -- *not* "didn't fall" alone, which the
  robot could trivially satisfy by leaning on a front foot, decaying the
  assist away before the front-leg-avoidance behavior it exists to teach had
  a chance to take hold (confirmed in a first attempt: assist reached 0 by
  iteration ~1145/5000 while ``front_contact_force`` had already plateaued).
- The commanded forward speed itself was too fast to hit without a bigger,
  faster weight-shift than 2 legs alone could yet manage -- narrowing
  ``lin_vel_x`` from (0.1, 1.0) down to (0.1, 0.3) (try16) is what finally
  produced a front-leg-free gait confirmed in play mode.

``front_contact_force`` is also raised from Phase 1's -0.6 to -0.9 (try9's
validated value) -- safe here because Phase 1 already carries the
``termination_penalty`` fix that makes any single penalty non-negotiable-but-
survivable rather than making early death cheaper than enduring it.

Revised (sandbox new-round Try-1) after the original forward-only, 0.1-0.3 m/s
version deployed to MuJoCo turned out to completely ignore the velocity
command. Root cause found in the training log: ``track_lin_vel_xy``'s ``std``
(``math.sqrt(0.25)`` ~= 0.5 m/s, inherited unchanged from Phase1's own
+-1.0 m/s range) is far wider than that 0.2 m/s-wide command range, so the
policy got a near-maximal tracking reward (~0.94) regardless of the actual
(much larger, ~0.4 m/s average) tracking error -- there was never a real
gradient pushing it to track the *specific* commanded speed, only to walk
forward at *some* pace. Rather than just narrowing ``std`` to match the
existing tiny range, the task itself was widened -- faster, and backward
walking added -- then ``std`` fixed to match:

- ``lin_vel_x``: (0.1, 0.3) forward-only -> +-(0.1, 0.4) magnitude, both
  signs, via ``ForwardAssistVelocityCommandCfg.lin_vel_x_min_magnitude``
  (samples ``sign * magnitude``, excluding the near-zero band from *both*
  directions -- a single continuous ``ranges.lin_vel_x`` interval can only
  exclude one whole sign, which is what forward-only walking did).
  ``ForwardAssistVelocityCommand._apply_assist_force`` was generalized to push
  in the *commanded* direction instead of forward-only, since the same
  front-leg-bracing risk applies symmetrically walking backward.
- ``lin_vel_y``: (-0.05, 0.05) -> exactly (0.0, 0.0) -- lateral movement
  disallowed entirely, not just narrowed.
- ``track_lin_vel_xy``/``track_ang_vel_z`` ``std``: tightened to 0.15 / 0.05,
  sized to the new (still much narrower than Phase1's +-1.0 m/s) ranges.

MuJoCo-confirmed working (responds to forward/backward commands) after this
change.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.biped import mdp
from unitree_rl_lab.tasks.biped.robots.go2.biped_env_cfg import FRONT_FOOT_NAMES, RobotEnvCfg, RobotPlayEnvCfg


def _apply_phase2_overrides(cfg):
    old = cfg.commands.base_velocity
    cfg.commands.base_velocity = mdp.ForwardAssistVelocityCommandCfg(
        asset_name=old.asset_name,
        resampling_time_range=old.resampling_time_range,
        rel_standing_envs=old.rel_standing_envs,
        heading_command=old.heading_command,
        debug_vis=old.debug_vis,
        ranges=old.ranges,
        front_foot_body_names=FRONT_FOOT_NAMES,
        state_file="logs/rsl_rl/go2_biped_phase2/assist_curriculum_state.json",
        lin_vel_x_min_magnitude=0.1,
    )
    cfg.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.4)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1)

    cfg.rewards.front_contact_force = RewTerm(
        func=mdp.front_foot_contact_force,
        weight=-0.9,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_FOOT_NAMES)},
    )
    cfg.rewards.track_lin_vel_xy.params["std"] = 0.15
    cfg.rewards.track_ang_vel_z.params["std"] = 0.05


@configclass
class CurriculumCfg:
    assist_force = CurrTerm(
        func=mdp.assist_force_decay,
        params={
            "command_name": "base_velocity",
            "success_threshold": 0.60,
            "decay_step": 0.01,
            "minimum_episodes": 1024,
            "front_contact_fraction_threshold": 0.10,
        },
    )


@configclass
class RobotEnvCfgPhase2(RobotEnvCfg):
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_phase2_overrides(self)


@configclass
class RobotPlayEnvCfgPhase2(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_phase2_overrides(self)
        # Play mode ignores the training curriculum's saved decay state and defaults
        # to no external assist -- edit initial_assist_scale below to manually test
        # with partial/full assist instead.
        self.commands.base_velocity.state_file = None
        self.commands.base_velocity.initial_assist_scale = 0.0
