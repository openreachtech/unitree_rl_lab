"""Go2W v2: v1 phase curricula with a teacher-style privileged critic.

Same terrain, rewards, commands, and TCN actor as ``Go2W-v1-Phase*``. The critic
observation is privileged ``xt`` only (foot-local height scans and contacts), and
``TcnTeacherPPORunnerCfg`` encodes ``xt`` then concatenates proprioception ``ot``.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg import ObservationsCfg
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase1 import (
    RobotEnvCfgPhase1,
    RobotPlayEnvCfgPhase1,
    RobotSceneCfgPhase1,
)
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase2 import (
    RobotEnvCfgPhase2,
    RobotPlayEnvCfgPhase2,
    RobotSceneCfgPhase2,
)
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase3 import (
    RobotEnvCfgPhase3,
    RobotPlayEnvCfgPhase3,
    RobotSceneCfgPhase3,
)
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase4 import (
    RobotEnvCfgPhase4,
    RobotPlayEnvCfgPhase4,
    RobotSceneCfgPhase4,
)
from unitree_rl_lab.tasks.locomotion.robots.go2w.velocity_env_cfg_phase5 import (
    RobotEnvCfgPhase5,
    RobotPlayEnvCfgPhase5,
    RobotSceneCfgPhase5,
)

# fmt: off
WHEEL_BODY_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
CALF_BODY_NAMES = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
THIGH_BODY_NAMES = ["FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"]
# fmt: on


def _foot_height_scanner(body_name: str) -> RayCasterCfg:
    """9-point height scan around a foot (3x3 grid, 10 cm spacing, 10 cm radius)."""
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + body_name,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.2, 0.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def _with_foot_scanners(base_cls: type) -> type:
    @configclass
    class Scene(base_cls):
        height_scanner_fr_foot = _foot_height_scanner("FR_foot")
        height_scanner_fl_foot = _foot_height_scanner("FL_foot")
        height_scanner_rr_foot = _foot_height_scanner("RR_foot")
        height_scanner_rl_foot = _foot_height_scanner("RL_foot")

    Scene.__name__ = base_cls.__name__.replace("RobotSceneCfg", "RobotSceneCfgV2", 1)
    Scene.__qualname__ = Scene.__name__
    return Scene


def _set_foot_scanner_periods(env_cfg) -> None:
    period = env_cfg.decimation * env_cfg.sim.dt
    for scanner in (
        env_cfg.scene.height_scanner_fr_foot,
        env_cfg.scene.height_scanner_fl_foot,
        env_cfg.scene.height_scanner_rr_foot,
        env_cfg.scene.height_scanner_rl_foot,
    ):
        scanner.update_period = period


RobotSceneCfgV2Phase1 = _with_foot_scanners(RobotSceneCfgPhase1)
RobotSceneCfgV2Phase2 = _with_foot_scanners(RobotSceneCfgPhase2)
RobotSceneCfgV2Phase3 = _with_foot_scanners(RobotSceneCfgPhase3)
RobotSceneCfgV2Phase4 = _with_foot_scanners(RobotSceneCfgPhase4)
RobotSceneCfgV2Phase5 = _with_foot_scanners(RobotSceneCfgPhase5)


@configclass
class ObservationsCfgV2(ObservationsCfg):
    """v1 policy observations, plus a teacher-style privileged critic group."""

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged ``xt`` for a teacher-style critic (not shown to the actor).

        Proprioception ``ot`` is the policy group; ``ActorCriticTcn`` encodes this
        group to a latent and concatenates ``ot`` before the value MLP.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )
        foot_height_scan_fr = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_fr_foot")},
            clip=(-1.0, 5.0),
        )
        foot_height_scan_fl = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_fl_foot")},
            clip=(-1.0, 5.0),
        )
        foot_height_scan_rr = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_rr_foot")},
            clip=(-1.0, 5.0),
        )
        foot_height_scan_rl = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_rl_foot")},
            clip=(-1.0, 5.0),
        )
        foot_contact_force = ObsTerm(
            func=mdp.contact_force_norm,
            scale=0.01,
            clip=(-100, 100),
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
                )
            },
        )
        foot_contact = ObsTerm(
            func=mdp.contact_states,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
                ),
                "threshold": 1.0,
            },
        )
        calf_contact = ObsTerm(
            func=mdp.contact_states,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=CALF_BODY_NAMES, preserve_order=True
                ),
                "threshold": 1.0,
            },
        )
        thigh_contact = ObsTerm(
            func=mdp.contact_states,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=THIGH_BODY_NAMES, preserve_order=True
                ),
                "threshold": 1.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class RobotEnvCfgV2Phase1(RobotEnvCfgPhase1):
    scene: RobotSceneCfgV2Phase1 = RobotSceneCfgV2Phase1(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgV2 = ObservationsCfgV2()

    def __post_init__(self):
        super().__post_init__()
        _set_foot_scanner_periods(self)


@configclass
class RobotPlayEnvCfgV2Phase1(RobotPlayEnvCfgPhase1):
    scene: RobotSceneCfgV2Phase1 = RobotSceneCfgV2Phase1(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgV2 = ObservationsCfgV2()

    def __post_init__(self):
        super().__post_init__()
        _set_foot_scanner_periods(self)


@configclass
class RobotEnvCfgV2Phase2(RobotEnvCfgPhase2):
    scene: RobotSceneCfgV2Phase2 = RobotSceneCfgV2Phase2(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgV2 = ObservationsCfgV2()

    def __post_init__(self):
        super().__post_init__()
        _set_foot_scanner_periods(self)


@configclass
class RobotPlayEnvCfgV2Phase2(RobotPlayEnvCfgPhase2):
    scene: RobotSceneCfgV2Phase2 = RobotSceneCfgV2Phase2(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgV2 = ObservationsCfgV2()

    def __post_init__(self):
        super().__post_init__()
        _set_foot_scanner_periods(self)


@configclass
class RobotEnvCfgV2Phase3(RobotEnvCfgPhase3):
    scene: RobotSceneCfgV2Phase3 = RobotSceneCfgV2Phase3(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgV2 = ObservationsCfgV2()

    def __post_init__(self):
        super().__post_init__()
        _set_foot_scanner_periods(self)


@configclass
class RobotPlayEnvCfgV2Phase3(RobotPlayEnvCfgPhase3):
    scene: RobotSceneCfgV2Phase3 = RobotSceneCfgV2Phase3(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgV2 = ObservationsCfgV2()

    def __post_init__(self):
        super().__post_init__()
        _set_foot_scanner_periods(self)


@configclass
class RobotEnvCfgV2Phase4(RobotEnvCfgPhase4):
    scene: RobotSceneCfgV2Phase4 = RobotSceneCfgV2Phase4(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgV2 = ObservationsCfgV2()

    def __post_init__(self):
        super().__post_init__()
        _set_foot_scanner_periods(self)


@configclass
class RobotPlayEnvCfgV2Phase4(RobotPlayEnvCfgPhase4):
    scene: RobotSceneCfgV2Phase4 = RobotSceneCfgV2Phase4(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgV2 = ObservationsCfgV2()

    def __post_init__(self):
        super().__post_init__()
        _set_foot_scanner_periods(self)


@configclass
class RobotEnvCfgV2Phase5(RobotEnvCfgPhase5):
    scene: RobotSceneCfgV2Phase5 = RobotSceneCfgV2Phase5(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgV2 = ObservationsCfgV2()

    def __post_init__(self):
        super().__post_init__()
        _set_foot_scanner_periods(self)


@configclass
class RobotPlayEnvCfgV2Phase5(RobotPlayEnvCfgPhase5):
    scene: RobotSceneCfgV2Phase5 = RobotSceneCfgV2Phase5(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgV2 = ObservationsCfgV2()

    def __post_init__(self):
        super().__post_init__()
        _set_foot_scanner_periods(self)
