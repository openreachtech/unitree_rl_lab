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
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.privileged import RingPatternCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_lidar import (
    GO2_LIDAR_SCANNER_CFG,
    LidarMapObsCfg,
    PLAY_LIDAR_HEIGHT_SCAN_CFG,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    CommandsCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
)

# ---------------------------------------------------------------------------
# Privileged critic input, after Lee et al. 2020 Table S4. See mdp/privileged.py for
# what the paper lists and which two rows are left out. Foot order is fixed here and
# shared by every per-foot term below, so the critic sees a consistent layout.
# ---------------------------------------------------------------------------
GO2_FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

def foot_ring_sensor(body_name: str) -> RayCasterCfg:
    """A 9-ray ring dropped from 20 m above one foot. 36 rays over four feet, negligible
    next to the body height scan's 187."""
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + body_name,
        # "yaw" so the ring stays gravity-aligned and keeps its 10 cm ground radius as the
        # foot pitches through the swing; "base" would tilt it and skew the sample spacing.
        ray_alignment="yaw",
        pattern_cfg=RingPatternCfg(radius=0.10, num_points=9),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

_FOOT_SCAN_CFGS = [SceneEntityCfg(f"foot_scan_{name}") for name in GO2_FEET]

CRITIC_HEIGHT_SCAN_CFG = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    clip=(-1.0, 5.0),
)


@configclass
class CriticCfgGo2(ObservationsCfg.CriticCfg):
    """Go2 critic: privileged ``height_scan`` plus Lee et al. 2020's foot-ground state.

    Temporal context is the GRU hidden state, so no term carries history.

    No body-centered ``height_scan`` here: Table S4 has no such term either, its whole
    terrain profile is the per-foot rings below. The grid is 187 numbers describing the
    terrain near the robot, which is largely redundant for a value function that only
    has to explain why the last steps went the way they did -- and the sensor feeding it
    is still in the scene for the terrain-adaptive foot-clearance rewards.
    """

    # Terrain profile at the feet rather than under the body -- these follow the swing
    # and land where contact happens. 36 + 12 dims.
    foot_height_scan = ObsTerm(
        func=mdp.foot_height_scan, params={"sensor_cfgs": _FOOT_SCAN_CFGS}, clip=(-1.0, 1.0)
    )
    foot_terrain_normal = ObsTerm(func=mdp.foot_terrain_normal, params={"sensor_cfgs": _FOOT_SCAN_CFGS})
    # Foot-ground interaction: which links are loaded, and how hard. 4 + 4 + 4 + 4 dims.
    foot_contact_force = ObsTerm(
        func=mdp.contact_force_magnitude,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=GO2_FEET)},
        scale=0.01,
    )
    foot_contact_state = ObsTerm(
        func=mdp.contact_states,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=GO2_FEET)},
    )
    thigh_contact_state = ObsTerm(
        func=mdp.contact_states,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh")},
    )
    calf_contact_state = ObsTerm(
        func=mdp.contact_states,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf")},
    )


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


# ---------------------------------------------------------------------------
# Play-only LiDAR view. This whole blind lineage exists to be the frozen controller
# under which a noisy-LiDAR height-map encoder is trained, so the thing worth watching
# in play is the fan-built map -- what that encoder will have to work from -- rather
# than the gait alone.
#
# Kept out of the training configs on purpose: the fan is 1080 rays that nothing in
# training reads, and the policy is blind by design. Attaching the sensor and the
# display-only observation group only in play keeps training exactly as it was.
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfgGo2LidarView(ObservationsCfgGo2):
    """Blind policy and privileged critic unchanged, with the fan-built grid alongside.

    Nothing reads ``lidar_map``; the observation manager computing it each step is what
    gives the term its chance to draw itself. See velocity_env_cfg_lidar.py.
    """

    lidar_map: LidarMapObsCfg = LidarMapObsCfg()


def apply_lidar_view(env_cfg) -> None:
    """Point a play config's LiDAR group at the visualising variant of the term.

    Call from ``__post_init__`` after ``super()``. The scene still has to declare
    ``lidar_scanner``; that is a class-level field, so each play config subclasses its
    phase's scene to add it.
    """
    env_cfg.observations.lidar_map.height_scan = PLAY_LIDAR_HEIGHT_SCAN_CFG
    # Only the fan-built grid on screen: the top-down scanner's own markers off.
    env_cfg.scene.height_scanner.debug_vis = False
