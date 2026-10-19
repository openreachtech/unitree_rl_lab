"""Anaguma walking expert: Tamura's Go2-Multitask-Gallop-Phase1/2 with only the robot swapped.

Same commands, rewards, terminations, events, terrain, curricula and unified observation as the Go2
tasks in ``tasks/locomotion/robots/go2/velocity_env_cfg_multitask.py`` -- that is the point: the
walking expert is copied, not re-designed. What has to change for a different machine:

  * ``scene.robot`` -> ``ANAGUMA_CFG`` (24.3 kg, legs 1.33x Go2's, see assets/robots/anaguma.py).
  * ``undesired_contacts`` drops ``Head_.*``: Anaguma has no head links, and a body pattern that
    resolves to nothing is an error.
  * The tow assist is scaled by the mass ratio (24.31 / 15.21 = 1.6) so it buys the same acceleration.
    A starting point, not a measurement -- assist strength has to be re-measured per condition.
  * ``apply_multitask_post_init`` reads ``actuators["GO2HV"]``; Anaguma's actuators are split into
    hip/thigh and calf groups, so the same tail is applied here over every actuator group. The
    shared file is left untouched.
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.anaguma import ANAGUMA_CFG
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_multitask import (
    RobotEnvCfgMultitaskGallopPhase1,
    RobotEnvCfgMultitaskGallopPhase2,
)
from unitree_rl_lab.tasks.multitask.robots.go2.multitask_env_cfg import (
    ACTUATOR_MAX_DELAY_STEPS,
    ACTUATOR_MIN_DELAY_STEPS,
)

# Whole-robot mass from each MJCF (Go2: unitree_mujoco go2.xml, Anaguma: NAS anaguma.xml).
ANAGUMA_MASS_RATIO = 24.31 / 15.21
LOG_ROOT = "logs/rsl_rl"


def apply_anaguma_post_init(cfg, log_name: str) -> None:
    """Swap Go2 for Anaguma after the Go2 config's own ``__post_init__`` has run."""
    cfg.scene.robot = ANAGUMA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    for actuator in cfg.scene.robot.actuators.values():
        actuator.min_delay = ACTUATOR_MIN_DELAY_STEPS
        actuator.max_delay = ACTUATOR_MAX_DELAY_STEPS

    cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]
    )

    # Own state files: the Go2 task's progress must not leak into this run, or the other way round.
    root = f"{LOG_ROOT}/{log_name}"
    tow = getattr(cfg.commands, "tow_assist", None)
    if tow is not None:
        tow.gain = tow.gain * ANAGUMA_MASS_RATIO
        tow.max_force = tow.max_force * ANAGUMA_MASS_RATIO
        if tow.state_file is not None:
            tow.state_file = f"{root}/tow_assist_state.json"
    if cfg.curriculum.lin_vel_cmd_levels.params.get("state_file") is not None:
        cfg.curriculum.lin_vel_cmd_levels.params["state_file"] = f"{root}/lin_vel_cmd_state.json"


def _play(cfg) -> None:
    cfg.scene.num_envs = 32
    cfg.scene.terrain.terrain_generator.num_rows = 3
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.commands.base_velocity.ranges = cfg.commands.base_velocity.limit_ranges
    tow = getattr(cfg.commands, "tow_assist", None)
    if tow is not None:
        tow.state_file = None
        tow.initial_assist_scale = 0.0
    cfg.curriculum.lin_vel_cmd_levels.params["state_file"] = None
    cfg.observations.policy.enable_corruption = False


@configclass
class AnagumaWalkPhase1EnvCfg(RobotEnvCfgMultitaskGallopPhase1):
    def __post_init__(self):
        super().__post_init__()
        apply_anaguma_post_init(self, "anaguma_multitask_gallop_phase1")


@configclass
class AnagumaWalkPhase1PlayEnvCfg(AnagumaWalkPhase1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class AnagumaWalkPhase2EnvCfg(RobotEnvCfgMultitaskGallopPhase2):
    def __post_init__(self):
        super().__post_init__()
        apply_anaguma_post_init(self, "anaguma_multitask_gallop_phase2")


@configclass
class AnagumaWalkPhase2PlayEnvCfg(AnagumaWalkPhase2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
