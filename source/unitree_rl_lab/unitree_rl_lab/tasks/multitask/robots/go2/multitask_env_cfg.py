"""Shared scene and unified observations for the Go2 multi-task tasks.

The locomotion and acrobatics policies were trained on different observation vectors -- 117/319 and
47/56 columns -- so neither can be dropped into a mixture of experts as it stands. This module
defines the superset both are expressed in: 122 columns for the actor, 330 for the critic. Every
term keeps the scale, noise and history it had in its source task, so a network trained here is
directly usable as an expert.

The **order** of the ``ObsTerm`` attributes below is load-bearing: Isaac Lab concatenates terms in
class-body declaration order, and ``obs_spec`` maps expert weights onto columns by that order. The
``assert_layout`` startup event re-checks it against the built managers, so a reordering fails at
construction instead of silently mis-feeding an expert.
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CORRECTED_CFG
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import EventCfg as LocomotionEventCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase1 import RobotSceneCfgPhase1
from unitree_rl_lab.tasks.multitask import mdp
from unitree_rl_lab.tasks.multitask.obs_spec import HISTORY_LENGTH

# The corrected actuator model -- the same one Go2-Jump-60, Go2-Run and Go2-Speed use, and the only
# one that has been checked against hardware. It differs from the stock model in three places, each
# closing a gap against the project's MuJoCo model: joint friction (Fs 0.2 / Fd 0.1 against none at
# all), a flat calf torque envelope with no speed derate, and armature 0.01 on every joint.
#
# This file previously took the stock model on the reasoning that an expert trained under one
# actuator model is not valid under the other, and that both source experts were trained under the
# stock one. That is true, and it is the cost rather than the argument: the stock model is the one
# whose sim2sim failures are on record -- a jump policy scoring 0.998 in Isaac Lab that pitches ~160
# deg and lands on its back in MuJoCo, and an armature mismatch that produced a full tumble. Keeping
# it means the merged policy is tuned against physics known to disagree with the robot it deploys
# to, which is what the MuJoCo runs of this task have been showing.
#
# Consequence to carry forward: this invalidates the *locomotion* expert too. Go2-Gallop reaches its
# robot through velocity_env_cfg's stock ROBOT_CFG, so it has to be retrained under this model
# before it can initialise the mixture, exactly as the acrobatics expert does.
ROBOT_CFG = UNITREE_GO2_CORRECTED_CFG


@configclass
class MultitaskSceneCfg(RobotSceneCfgPhase1):
    """Flat-terrain scene carrying every sensor either source task needs.

    Taken from the locomotion side rather than the acrobatics side: it already has the
    ``height_scanner`` the locomotion critic reads, and its flat *generator* terrain is what the
    locomotion policy was trained on. The acrobatics task used a bare ``plane``, which is
    equivalent ground for a flat-only task but would leave the height scanner without a mesh to
    cast against -- and would put the acrobatics re-training on different ground from the merged
    environment it is meant to feed.
    """

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class UnifiedObservationsCfg:
    """The 122-column actor / 330-column critic superset.

    Term order must match ``obs_spec.POLICY_UNIFIED`` and ``obs_spec.CRITIC_UNIFIED``.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # -- shared by both source tasks (identical scale and noise in each, verified)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        # -- locomotion task selector
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        # -- acrobatics task selector: (enabled, target_height, pitch_turns, roll_turns) + phase
        jump_command = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "jump"})
        jump_time = ObsTerm(func=mdp.jump_time_encoding, clip=(-100, 100), params={"command_name": "jump"})
        # -- proprioception, with the locomotion side's history (the acrobatics task had none;
        #    the extra frames are what let one network serve a gait and a flip)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=HISTORY_LENGTH,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=HISTORY_LENGTH,
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100), history_length=HISTORY_LENGTH)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        jump_command = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "jump"})
        jump_time = ObsTerm(func=mdp.jump_time_encoding, clip=(-100, 100), params={"command_name": "jump"})
        # -- privileged state the acrobatics critic was trained on
        root_height = ObsTerm(func=mdp.root_height, clip=(-100, 100))
        root_roll_angle = ObsTerm(func=mdp.root_roll_angle, clip=(-100, 100))
        root_pitch_angle = ObsTerm(func=mdp.root_pitch_angle, clip=(-100, 100))
        maximum_jump_height = ObsTerm(func=mdp.maximum_jump_height, clip=(-100, 100), params={"command_name": "jump"})
        accumulated_root_pitch = ObsTerm(
            func=mdp.accumulated_root_pitch, clip=(-100, 100), params={"command_name": "jump"}
        )
        accumulated_root_roll = ObsTerm(
            func=mdp.accumulated_root_roll, clip=(-100, 100), params={"command_name": "jump"}
        )
        # -- proprioception
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), history_length=HISTORY_LENGTH)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), history_length=HISTORY_LENGTH)
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100), history_length=HISTORY_LENGTH)
        # -- privileged terrain the locomotion critic was trained on
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


def apply_multitask_post_init(cfg) -> None:
    """Shared ``__post_init__`` tail for every multi-task environment.

    Both source tasks already run at 200 Hz physics with a decimation of 4 (a 50 Hz control rate)
    and the same action scale, which is what makes a weighted sum of their actions meaningful --
    this keeps that fixed in one place rather than in each config.
    """
    cfg.decimation = 4
    cfg.sim.dt = 0.005
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physics_material = cfg.scene.terrain.physics_material
    cfg.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
    cfg.scene.contact_forces.update_period = cfg.sim.dt
    cfg.scene.height_scanner.update_period = cfg.decimation * cfg.sim.dt
    # Mirror the locomotion base config: the terrain generator's curriculum has to be on exactly
    # when a terrain_levels curriculum term exists, since that term promotes environments between
    # difficulty rows the generator only builds when asked. The acrobatics family has no such term
    # (its curriculum only decays the assist force), the locomotion family does.
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = getattr(cfg.curriculum, "terrain_levels", None) is not None




@configclass
class MultitaskEventCfg(LocomotionEventCfg):
    """The event set every multi-task environment shares -- experts and merged policy alike.

    Defined once because getting it wrong is invisible until it is expensive. The merged environment
    originally took this set on the reasoning that it was "the more demanding of the two", which
    quietly exposed the acrobatics expert to two disturbances it had never trained against:
    ``add_base_mass`` adds up to +3 kg to a ~15 kg robot (a fixed impulse then buys proportionally
    less height), and ``push_robot`` fires a velocity disturbance every 5-10 s, which in a 20 s
    episode can land mid-flip. Measured jump height sat at 0.016 m against the 0.10 m the expert
    reaches on its own.

    The fix is not to strip the disturbances out of the merged environment -- that would throw away
    robustness the locomotion expert already has, and the real robot's mass and contacts are not
    ideal either. It is to train the acrobatics expert against them too, so both halves arrive
    equally hardened. Sharing one config is what stops the two from drifting apart again.
    """
