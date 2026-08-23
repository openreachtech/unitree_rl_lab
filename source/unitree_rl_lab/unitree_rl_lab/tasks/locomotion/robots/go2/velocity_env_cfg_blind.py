"""Blind Go2: proprioception-only actor, privileged critic.

Ported verbatim from ``robots/go2/velocity_env_cfg_go2.py`` as it stands on
``feat/go2-curriculum`` (the ``Unitree-Go2-Velocity-v1`` lineage), with only the
intra-package imports rewritten. That version predates the height scan being added to
the *policy* group, so the actor sees proprioception alone while the critic keeps
``CRITIC_HEIGHT_SCAN_CFG`` -- the asymmetric actor-critic this whole line depends on.
Kept as a copy rather than reusing ``velocity_env_cfg_go2.py`` in this same package,
because that module has since grown the sighted policy, and the point here is a policy
that never sees terrain. Class names are left as the v1 originals so this file still
diffs cleanly against ``feat/go2-curriculum``; the ``velocity_env_cfg_blind_*`` file
names are what keep it apart from the sighted configs alongside it.

The tasks built on this are registered with ``GruPPORunnerCfg``, the only other
difference from the v1 runs: a GRU in front of the actor and critic MLPs, giving the
blind policy the memory it needs to infer terrain from how the last few steps went.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    CommandsCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
)

CRITIC_HEIGHT_SCAN_CFG = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    clip=(-1.0, 5.0),
)


@configclass
class CriticCfgGo2(ObservationsCfg.CriticCfg):
    """Go2 critic: privileged ``height_scan``. Temporal context is the GRU hidden state."""

    height_scan = CRITIC_HEIGHT_SCAN_CFG


@configclass
class ObservationsCfgGo2(ObservationsCfg):
    """Go2 observations: proprioception-only policy; privileged critic with height scan."""

    critic: CriticCfgGo2 = CriticCfgGo2()


@configclass
class CommandsCfgGo2(CommandsCfg):
    """Go2 v1: reduce standing-only env fraction."""

    base_velocity = CommandsCfg().base_velocity.replace(rel_standing_envs=0.01)


@configclass
class RewardsCfgGo2(RewardsCfg):
    """Go2-specific reward tuning."""

    track_ang_vel_z = RewardsCfg().track_ang_vel_z.replace(weight=1.0)

    wild_foot_clearance = RewTerm(
        func=mdp.wild_foot_clearance_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": 0.4,
            "offset": [0.0, 0.5, 0.5, 0.0],  # trot: FR+RL vs FL+RR
            "radius": 0.1,
            "target_clearance": 0.05,
        },
    )

    foot_clearance_terrain_adaptive = RewTerm(
        func=mdp.foot_clearance_terrain_adaptive,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "target_clearance": 0.05,
            "command_name": "base_velocity",
        },
    )

    forward_command_progress = RewTerm(
        func=mdp.forward_command_progress,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RobotEnvCfgGo2(RobotEnvCfg):
    """Shared Go2 v1 MDP settings."""

    observations: ObservationsCfgGo2 = ObservationsCfgGo2()
    commands: CommandsCfgGo2 = CommandsCfgGo2()
    rewards: RewardsCfgGo2 = RewardsCfgGo2()
