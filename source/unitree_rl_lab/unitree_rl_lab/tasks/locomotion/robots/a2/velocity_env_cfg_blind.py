"""Blind A2: proprioception-only actor, privileged critic.

The A2 counterpart of ``robots/go2/velocity_env_cfg_blind.py``, structurally identical:
the actor sees 45 proprioceptive numbers and nothing else, while the critic keeps
Lee et al. 2020's foot-ground state, and the tasks built on this run a GRU in front of
both MLPs (``GruPPORunnerCfg``) so the policy can infer terrain from how the last few
steps went.

The Go2 module also carries a play-only LiDAR fan; nothing here does. The A2 lineage
starts blind and stays blind, and the fan can be added later against A2's own
``front_lidar_link`` / ``rear_lidar_link`` mounts, which ``a2_description`` already
places at (0.33767, 0, 0.08134) and behind it.

Lengths are Go2's x1.29 and gait times Go2's x1.14; see ``velocity_env_cfg.py`` for
where the two factors come from.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.privileged import RingPatternCfg
from unitree_rl_lab.tasks.locomotion.robots.a2.velocity_env_cfg import (
    CommandsCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
)

# ---------------------------------------------------------------------------
# Gait clock, shared by every phase-gated term in this lineage so they cannot drift
# apart. Go2's 0.4 s x1.14 (Froude): a swing is a pendulum half-period, and A2's leg is
# 1.29x longer. Go2 repeats the literal at each call site; one constant here is the same
# recipe with the drift removed.
# ---------------------------------------------------------------------------
A2_GAIT_PERIOD = 0.46

A2_TROT_OFFSET = [0.0, 0.5, 0.5, 0.0]
"""Trot: feet 0 and 3 in phase, 1 and 2 half a cycle later.

These index the *articulation's* body order, not the order written in the ``body_names``
list -- ``SceneEntityCfg`` resolves without preserving order, so the list at each call
site is only a filter. Measured on the imported A2, that order is
``FL_foot, FR_foot, RL_foot, RR_foot``, the same as Go2's, which makes 0 and 3 the FL/RR
diagonal and 1 and 2 the FR/RL one -- a real trot. (It is *not* the order
``a2_description`` declares its legs in, FL, RL, FR, RR; the URDF importer reorders.)
Worth re-checking against ``robot.data.body_names`` if the asset is ever rebuilt.
"""

# ---------------------------------------------------------------------------
# Privileged critic input, after Lee et al. 2020 Table S4. See mdp/privileged.py for
# what the paper lists and which two rows are left out. Foot order is fixed here and
# shared by every per-foot term below, so the critic sees a consistent layout.
# ---------------------------------------------------------------------------
A2_FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def foot_ring_sensor(body_name: str) -> RayCasterCfg:
    """A 9-ray ring dropped from 20 m above one foot. 36 rays over four feet, negligible
    next to the body height scan's 609.

    Radius 0.13 m is Go2's 0.10 x1.29 -- the ring has to stay the same fraction of a
    stride so a "local" terrain sample means the same thing on both robots.
    """
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + body_name,
        # "yaw" so the ring stays gravity-aligned and keeps its ground radius as the foot
        # pitches through the swing; "base" would tilt it and skew the sample spacing.
        ray_alignment="yaw",
        pattern_cfg=RingPatternCfg(radius=0.13, num_points=9),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


_FOOT_SCAN_CFGS = [SceneEntityCfg(f"foot_scan_{name}") for name in A2_FEET]

@configclass
class CriticCfgA2(ObservationsCfg.CriticCfg):
    """A2 critic: Lee et al. 2020's foot-ground state, no body-centered height scan.

    Temporal context is the GRU hidden state, so no term carries history.

    Table S4 has no body-centred terrain term either -- its whole terrain profile is the
    per-foot rings below, which follow the swing and land where contact happens. The
    top-down scanner stays in the scene because the terrain-adaptive clearance rewards
    read it.
    """

    # Terrain profile at the feet rather than under the body. 36 + 12 dims.
    foot_height_scan = ObsTerm(
        func=mdp.foot_height_scan, params={"sensor_cfgs": _FOOT_SCAN_CFGS}, clip=(-1.0, 1.0)
    )
    foot_terrain_normal = ObsTerm(func=mdp.foot_terrain_normal, params={"sensor_cfgs": _FOOT_SCAN_CFGS})
    # Foot-ground interaction: which links are loaded, and how hard. 4 + 4 + 4 + 4 dims.
    #
    # scale 0.01 / 2.67: A2 weighs 40 kg against Go2's 15, so a stance foot carries
    # 2.67x the load. At Go2's scale the term would enter the critic that much louder
    # than its neighbours; at 0.0037 a two-foot stance reads about 0.75 on both robots.
    foot_contact_force = ObsTerm(
        func=mdp.contact_force_magnitude,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=A2_FEET)},
        scale=0.0037,
    )
    foot_contact_state = ObsTerm(
        func=mdp.contact_states,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=A2_FEET)},
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
class ObservationsCfgA2(ObservationsCfg):
    """A2 observations: proprioception-only policy; privileged critic."""

    critic: CriticCfgA2 = CriticCfgA2()


@configclass
class CommandsCfgA2(CommandsCfg):
    """Reduce the standing-only env fraction, as the Go2 lineage does."""

    base_velocity = CommandsCfg().base_velocity.replace(rel_standing_envs=0.01)


@configclass
class RewardsCfgA2(RewardsCfg):
    """A2-specific reward tuning.

    The three clearance/progress terms sit at weight 0 here and are switched on by the
    phase that needs them, exactly as on Go2. Their length parameters are Go2's x1.29
    and their periods x1.14.
    """

    track_ang_vel_z = RewardsCfg().track_ang_vel_z.replace(weight=1.0)

    wild_foot_clearance = RewTerm(
        func=mdp.wild_foot_clearance_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "period": A2_GAIT_PERIOD,
            "offset": A2_TROT_OFFSET,
            # x1.29 both: the radius that counts as "under this foot", and the lift that
            # counts as cleared.
            "radius": 0.13,
            "target_clearance": 0.065,
        },
    )

    foot_clearance_terrain_adaptive = RewTerm(
        func=mdp.foot_clearance_terrain_adaptive,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "target_clearance": 0.065,
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
class RobotEnvCfgA2(RobotEnvCfg):
    """Shared A2 blind-lineage MDP settings."""

    observations: ObservationsCfgA2 = ObservationsCfgA2()
    commands: CommandsCfgA2 = CommandsCfgA2()
    rewards: RewardsCfgA2 = RewardsCfgA2()
