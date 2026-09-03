"""Go2-Crouch-Phase1: walk while tracking a commanded base height, on an infinite flat plane.

Based on Unitree-Go2-Velocity-v1-Phase1 (same small velocity-command ranges -- walking
should stay slow while crouched, this isn't about speed) with these changes:

- ``base_height`` is a command term (``UniformHeightCommand``), not a reward constant, so the
  target can differ per env / be widened by a curriculum without touching the reward function.
- That curriculum is ``crouch_depth_levels``: ``base_height.ranges`` starts degenerate at
  ``STANDING_HEIGHT`` (0 crouch depth -- learn to walk normally first, same bootstrapping
  idea as Phase1) and its lower bound deepens by 1cm every time ``track_base_height`` clears
  80% of its max weight, toward ``MAX_CROUCH_DEPTH`` -- exactly how ``lin_vel_cmd_levels``
  widens ``base_velocity`` toward its own ``limit_ranges``, and on the same schedule (both
  curriculum terms gate on the same per-reset step, so speed and crouch depth ramp up
  together rather than one waiting for the other).
- Terrain is an infinite plane (``terrain_type="plane"``), not Phase1's generated flat grid
  -- cheaper, and height-tracking doesn't need terrain variety or a terrain_levels curriculum.

Two fixed values a fixed-height try established:
- ``STANDING_HEIGHT = 0.4``: UNITREE_GO2_CFG.init_state.pos.z (the Go2's stock standing/spawn
  height -- not 0.32m).
- ``MAX_CROUCH_DEPTH = 0.25``: a fixed 0.2m target (20cm crouch) measured out at ~0.226m
  actual under a flat exp-kernel tracking reward -- undershooting the command by ~2.6cm.
  0.25m is a deliberately deeper ceiling than what 0.2m alone achieved, not a claim that
  0.25m itself will be reached exactly -- the curriculum simply stops deepening wherever
  tracking quality actually caps out.

1000-iteration run (2026-09-01): crouch_depth_levels reached a 0.40 -> 0.30 lower bound
(10cm of the 25cm ceiling) by iteration 900, then stalled -- the flat tracking reward settled
just under the curriculum's 80% gate, so widening self-throttled there. Play afterward,
sampling the *full* [0.15, 0.40] range (see RobotPlayEnvCfgCrouchPhase1), showed barely any
visible crouching: most sampled depths were below the 0.30m frontier training had actually
reached, i.e. out of distribution for what the policy had practiced. A second 1000-iteration
run with lin_vel_x's limit cut 1.5 -> 1.0 (see CommandsCfgCrouchPhase1 below) reproduced the
exact same 0.30m plateau, ruling out "competing with top speed" as the bottleneck.

Working theory: the flat tracking reward's pull toward the target was exactly as strong at
a 25cm crouch as at 0, but penalties like joint_torques_l2/energy get relatively more
expensive the deeper the crouch (holding a low base height costs more static torque almost
by physics) -- an increasingly lopsided trade that gets worse, not better, as depth
increases. Three sandbox tries attacked this (see mdp/rewards.py's depth-scaled reward
docstring): a flat weight bump reached 19cm but with unstable tracking error late in
training; discounting joint_torques/energy by depth reached 16cm and was still improving,
not yet plateaued; **``track_base_height_depth_scaled_exp`` below -- weight 1.0->3.0 AND
std 0.05->0.10 both scaling with depth -- reached the full 25cm ceiling (0.15m) by iteration
1300 of a 2000-iteration run and held it stably through the end, with base_height error
improving to ~0.024m.** Confirmed in Play. Promoted here as the default reward; the other
two tries were deleted after losing the comparison.

This is Phase1: a single flat-terrain task to validate that base-height tracking actually
produces a crouched gait before building anything else on top of it. Phase2 (see
velocity_env_cfg_crouch_phase2.py) adds terrain variety (rough ground + boxes).

(Renamed from Go2-Crouch/velocity_env_cfg_crouch.py to Go2-Crouch-Phase1 on 2026-09-02, when
Phase2 was introduced -- no functional changes, only the Crouch* -> CrouchPhase1 class/id
rename.)
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg, RobotSceneCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import (
    CommandsCfgGo2,
    CriticCfgGo2,
    ObservationsCfgGo2,
    PolicyCfgGo2,
    RewardsCfgGo2,
    RobotEnvCfgGo2,
)

STANDING_HEIGHT = 0.4
MAX_CROUCH_DEPTH = 0.25


@configclass
class RobotSceneCfgCrouchPhase1(RobotSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )


@configclass
class CommandsCfgCrouchPhase1(CommandsCfgGo2):
    base_velocity = CommandsCfgGo2().base_velocity.replace(
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)
        ),
        # lin_vel_x capped at 1.0 (was 1.5): the first curriculum run showed speed maxing
        # out around the same time crouch_depth_levels stalled at 10cm of 25cm -- easing the
        # speed ceiling should leave more of track_base_height's headroom for depth instead
        # of fast walking. lin_vel_y is untouched: already 0.8, under the new 1.0 cap.
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.2, 1.2)
        ),
    )

    # Starts degenerate at STANDING_HEIGHT (0 crouch depth) -- crouch_depth_levels deepens
    # the lower bound toward limit_ranges as track_base_height clears its threshold.
    base_height = mdp.UniformLevelHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=mdp.UniformHeightCommandCfg.Ranges(base_height=(STANDING_HEIGHT, STANDING_HEIGHT)),
        limit_ranges=mdp.UniformHeightCommandCfg.Ranges(
            base_height=(STANDING_HEIGHT - MAX_CROUCH_DEPTH, STANDING_HEIGHT)
        ),
    )


@configclass
class PolicyCfgCrouchPhase1(PolicyCfgGo2):
    base_height_command = ObsTerm(
        func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_height"}
    )


@configclass
class CriticCfgCrouchPhase1(CriticCfgGo2):
    base_height_command = ObsTerm(
        func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_height"}
    )


@configclass
class ObservationsCfgCrouchPhase1(ObservationsCfgGo2):
    policy: PolicyCfgCrouchPhase1 = PolicyCfgCrouchPhase1()
    critic: CriticCfgCrouchPhase1 = CriticCfgCrouchPhase1()


@configclass
class RewardsCfgCrouchPhase1(RewardsCfgGo2):
    # Promoted from sandbox Try 2 (2026-09-02): depth-scaled exp-kernel, not a flat
    # track_base_height_exp. At standing height (depth_frac=0): weight_scale=1.0, std=0.05
    # -- what the original flat version used everywhere. At the deepest commandable crouch
    # (depth_frac=1): weight_scale=3.0, std=0.10 -- 3x the pull, 2x the error tolerance.
    # Reached the full 25cm ceiling and held it stably where the flat version plateaued at
    # 10cm -- see module docstring.
    track_base_height = RewTerm(
        func=mdp.track_base_height_depth_scaled_exp,
        weight=1.0,
        params={
            "command_name": "base_height",
            "standing_height": STANDING_HEIGHT,
            "max_depth": MAX_CROUCH_DEPTH,
            "std_min": 0.05,
            "std_max": 0.10,
            "weight_min": 1.0,
            "weight_max": 3.0,
        },
    )

    # joint_position_penalty pulls joints back toward the *standing* default_joint_pos
    # (scaled up 5x whenever base_velocity is near zero) -- directly fighting the crouched
    # posture track_base_height above is trying to teach. Off for this task; re-enable/tune
    # once crouched walking is established and stand-still stability needs revisiting.
    joint_pos = RewardsCfgGo2().joint_pos.replace(weight=0.0)


@configclass
class CurriculumCfgCrouchPhase1(CurriculumCfg):
    """Infinite plane, not a generated terrain grid -- no levels for terrain_levels_vel."""

    terrain_levels = None
    crouch_depth_levels = CurrTerm(func=mdp.crouch_depth_levels)


@configclass
class RobotEnvCfgCrouchPhase1(RobotEnvCfgGo2):
    """Go2-Crouch-Phase1: walk while tracking a commanded base height, curriculum-widened
    from standing height down toward MAX_CROUCH_DEPTH by crouch_depth_levels."""

    scene: RobotSceneCfgCrouchPhase1 = RobotSceneCfgCrouchPhase1(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgCrouchPhase1 = ObservationsCfgCrouchPhase1()
    commands: CommandsCfgCrouchPhase1 = CommandsCfgCrouchPhase1()
    rewards: RewardsCfgCrouchPhase1 = RewardsCfgCrouchPhase1()
    curriculum: CurriculumCfgCrouchPhase1 = CurriculumCfgCrouchPhase1()


@configclass
class RobotPlayEnvCfgCrouchPhase1(RobotEnvCfgCrouchPhase1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Play evaluates across the *full* commandable range, not wherever training's
        # curriculum frontier happened to reach -- see module docstring for why that gap
        # is exactly what made the trained policy look like it "doesn't crouch" in Play.
        self.commands.base_height.ranges = self.commands.base_height.limit_ranges
