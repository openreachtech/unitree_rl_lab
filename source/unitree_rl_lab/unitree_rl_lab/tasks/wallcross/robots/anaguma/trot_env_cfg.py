"""Anaguma ordinary walking expert (trot) on the unified multi-task observation.

The gallop expert in ``walk_env_cfg.py`` copies Tamura's Go2-Multitask-Gallop exactly, and so it
gallops/bounds forward at every speed (measured 2026-10-14: even at 0.5 m/s the front pair lands
0.1 cycle apart and the hind pair 0.1-0.2 later; backward/lateral trot). The walking expert wanted
for the wall-crossing pipeline is an ordinary walk, so this task starts from the plain Go2 walking
setup instead (``Unitree-Go2-Velocity-v1-Phase1``: flat ground, ``RewardsCfgGo2``) and keeps only
what the multi-task pipeline needs:

  * the unified observation (actor 124 / critic 335), idle jump and handstand commands and the
    shared multi-task events -- the same wrapping ``velocity_env_cfg_multitask.py`` applies, so
    the expert still drops into the mixture-of-experts policy;
  * the Anaguma robot and its fixes (``apply_anaguma_post_init``).

Gallop-specific parts are left out: no ``paired_gait``, no tow assist, and the reward weights the
gallop loosened for bounce (``base_linear_velocity``, ``flat_orientation_l2``, ``feet_air_time``,
``air_time_variance``) stay at the walking values. ``diagonal_trot_reward`` asks for a trot
explicitly. The command range is Phase1's walking range (forward/backward 1.5, lateral 0.8,
turning 1.2), widened by the velocity ratchet as tracking improves, with a way back down.

Not yet trained (GPU busy on 2026-10-14). Untuned: the trot weight (0.5) is a starting value.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import RewardsCfgGo2
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase1 import (
    CommandsCfgPhase1,
    RobotEnvCfgPhase1,
)
from unitree_rl_lab.tasks.multitask.robots.go2.multitask_env_cfg import (
    IDLE_HANDSTAND_COMMAND,
    IDLE_JUMP_COMMAND,
    MultitaskEventCfg,
    MultitaskSceneCfg,
    UnifiedObservationsCfg,
)

from .walk_env_cfg import _play, apply_anaguma_post_init
from .walk_rewards import diagonal_trot_reward

LOG_NAME = "anaguma_multitask_trot"
_FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


@configclass
class TrotCommandsCfg(CommandsCfgPhase1):
    jump = IDLE_JUMP_COMMAND
    handstand = IDLE_HANDSTAND_COMMAND


@configclass
class TrotRewardsCfg(RewardsCfgGo2):
    trot = RewTerm(
        func=diagonal_trot_reward,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FEET, preserve_order=True)},
    )


@configclass
class TrotCurriculumCfg(CurriculumCfg):
    # Two-way ratchet with its own state file, the same protection the gallop curriculum adds
    # (a one-way ratchet can push the range past what the robot can do and never come back).
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={"decrease_threshold": 0.6, "state_file": f"logs/rsl_rl/{LOG_NAME}/lin_vel_cmd_state.json"},
    )


@configclass
class AnagumaTrotEnvCfg(RobotEnvCfgPhase1):
    scene: MultitaskSceneCfg = MultitaskSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: UnifiedObservationsCfg = UnifiedObservationsCfg()
    commands: TrotCommandsCfg = TrotCommandsCfg()
    events: MultitaskEventCfg = MultitaskEventCfg()
    rewards: TrotRewardsCfg = TrotRewardsCfg()
    curriculum: TrotCurriculumCfg = TrotCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # The tail of apply_multitask_post_init, minus its GO2HV lookup (done per actuator group
        # in apply_anaguma_post_init).
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.terrain.terrain_generator.curriculum = True
        apply_anaguma_post_init(self, LOG_NAME)


@configclass
class AnagumaTrotPlayEnvCfg(AnagumaTrotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
