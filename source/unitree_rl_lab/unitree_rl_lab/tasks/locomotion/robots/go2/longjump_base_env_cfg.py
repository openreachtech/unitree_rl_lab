"""Stage A of the running long jump task ("Unitree-Go2-LongJump-Base-v1").

Trains a flat-ground running policy from scratch with an extended forward
speed ceiling, to be used as the Net2Net base for Stage B
("Unitree-Go2-LongJump-v1", jump-integrated). See project memory:
- プロジェクト_走り幅跳び理論値計算.md (theoretical distance targets)
- リファレンス_キーボード操作とNet2Net設計の下調べ.md (Stage A/B split rationale,
  why the existing unitree_go2_velocity_v1 checkpoint was NOT reused: its
  terrain history is ambiguous -- possibly rough/stairs via
  GO2_CURRICULUM_TERRAIN_CFG -- and this task wants a clean flat-ground
  policy for a theoretical-limit measurement)
- プロジェクト_StageB崩壊の原因分析.md (the 2026-09-02 reward/actuator/DR rework
  below, and the measurements behind each change)

Deliberately unchanged from the upstream ``RobotEnvCfg``:
- Terrain: ``COBBLESTONE_ROAD_CFG`` already resolves to 100% flat ground
  (every sub-terrain except "flat" is commented out there), so no terrain
  override is needed for a flat-ground task.
- Observations: the base ``ObservationsCfg`` has no height_scan on either
  the policy or critic group already, matching the blind-policy decision
  for this task family.
"""

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots import unitree_actuators
from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    CommandsCfg,
    EventCfg,
    RewardsCfg,
    RobotEnvCfg,
)

# --- corrected actuator model (2026-09-02) -------------------------------------------
# Ported from tak's Go2-Jump-60 (tag ``jump-demo``), where it was established against
# the project's own MuJoCo model rather than guessed. Two things were wrong for a
# jumping task:
#
# TORQUE: the stock UnitreeActuatorCfg_Go2HV gives every joint the bare GO-M8010-6 curve
# (Y1 20.2 / Y2 23.4 N*m). That is right for hip and thigh, which sit on the motor, and
# wrong for the calf (knee), which reaches roughly double -- go2.xml clamps the knee at
# +/-45.43. Modelling the knee at half its real strength is itself a sufficient reason
# for a policy to never use it: tak's MuJoCo capture of a jump measured the knee doing
# 21 J against the thigh's 60 J, i.e. acting as a strut rather than a motor, with the
# thigh (the *weaker* joint) saturating for 198 ms of a 351 ms push-off. A long jump's
# take-off is exactly the manoeuvre that needs the knee.
#
# ARMATURE / FRICTION: matched to MuJoCo's 0.01 and frictionloss 0.2 / damping 0.1, so
# the two simulators agree on joint inertia and losses. Leaving PhysX's own `friction`
# at 0.01 alongside Fs=0.2 would double-count what MuJoCo models once, so it goes to 0.
CALF_PEAK_TORQUE = 45.43
CALF_PUSH_TORQUE = 45.43 * (20.2 / 23.4)  # keep the stock Y1/Y2 ratio -> 39.22

GO2_CORRECTED_ACTUATOR_CFG: ArticulationCfg = UNITREE_GO2_CFG.replace(
    actuators={
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
            Fs=0.2,
            Fd=0.1,
            Y1={
                ".*_hip_joint": 20.2,
                ".*_thigh_joint": 20.2,
                ".*_calf_joint": CALF_PUSH_TORQUE,  # 39.22
            },
            Y2={
                ".*_hip_joint": 23.4,
                ".*_thigh_joint": 23.4,
                ".*_calf_joint": CALF_PEAK_TORQUE,  # 45.43
            },
            X1={".*_hip_joint": 13.5, ".*_thigh_joint": 13.5, ".*_calf_joint": 13.5},
            X2={".*_hip_joint": 30.0, ".*_thigh_joint": 30.0, ".*_calf_joint": 30.0},
            armature={".*_hip_joint": 0.01, ".*_thigh_joint": 0.01, ".*_calf_joint": 0.01},
        ),
    },
)


@configclass
class CommandsCfgLongJumpBase(CommandsCfg):
    """Same as the base velocity command, with a higher forward-speed ceiling.

    2026-08-31: forward-only (no backward at all), after mujoco sim2sim
    testing on the first Stage A/B pair showed the robot walks backward
    fine but falls immediately when commanded forward -- the symmetric
    (-1.0, 5.5) range let lin_vel_cmd_levels' combined track_lin_vel_xy
    gate get satisfied by backward walking alone (an easier gait for this
    morphology, apparently) while forward locomotion stayed undertrained.
    Clamping both the curriculum ceiling and the initial sampling range's
    floor to 0.0 removes that escape hatch entirely -- every sampled
    command is forward-or-standing, never backward. See
    フィードバック_ActorCriticEEのONNXエクスポートバグ.md and
    プロジェクト_StageA完了とStageB移行.md for the mujoco test that surfaced this.
    """

    base_velocity = CommandsCfg().base_velocity.replace(
        ranges=CommandsCfg().base_velocity.ranges.replace(lin_vel_x=(0.0, 0.1)),
        limit_ranges=CommandsCfg().base_velocity.limit_ranges.replace(lin_vel_x=(0.0, 5.5)),
    )


@configclass
class RewardsCfgLongJumpBase(RewardsCfg):
    """Reward budget rebalanced so that staying alive beats terminating.

    2026-09-02, from the Stage B v2 collapse post-mortem
    (プロジェクト_StageB崩壊の原因分析.md). Measured at the moment the policy
    collapsed (iteration 1400, lin_vel_cmd_levels 2.2 m/s), per second of episode:

        positives   track_lin_vel_xy +1.079   track_ang_vel_z +0.467   (cap 2.25)
        negatives   action_rate      -1.312   jump_dense      -0.558
                    joint_pos        -0.537   joint_acc/vel/torques -0.568
                    everything else  -0.485   TOTAL           -3.460
        net         -1.85 per second  =>  -39.7 return per 18.5 s episode

    Meanwhile a policy that fell over immediately scored **+6.0** per 0.55 s episode.
    Dying was worth ~+46 return, so PPO collapsed to it -- entropy (13.5) and action
    noise (0.77) were unchanged across the collapse, i.e. this was correct optimisation
    of a misspecified objective, not an exploration failure.

    Two structural problems and their fixes:

    1. The positive terms are bounded by their weights (2.25/s at best) while the
       penalties are L2 in joint velocity / acceleration / action rate and therefore
       grow with the SQUARE of commanded speed. Beyond roughly 1 m/s of commanded
       speed no policy, however good, can score positive. ``alive`` and ``upright``
       add a speed-independent positive baseline (+3.0/s), following tak's working
       Go2-Jump reward set (upright 2.0 / standing_pose 1.0 / stillness 1.0).
    2. ``action_rate`` alone was 38% of the entire penalty budget, and it opposes
       exactly the explosive extension a jump needs. Cut 10x, matching tak's
       -0.1 -> -0.01 for the same reason. ``joint_torques`` is L2 so it charges the
       knee 5.1x more than the thigh at capacity (2025 vs 400 units) -- backwards for
       a task that needs the knee -- cut 10x rather than removed outright.
       ``joint_pos`` cut 7x: at -0.537/s it was the third largest penalty and it
       pulls toward the default standing pose, which a running crouch is not.
    """

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    upright = RewTerm(func=mdp.upright_reward, weight=2.0, params={"std": 0.25})

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-5)
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )


@configclass
class EventCfgLongJump(EventCfg):
    """Base events plus the domain randomisation the sim2sim gap called for.

    2026-09-02: the base config randomises friction and adds base mass, but leaves
    per-body mass and PD gains fixed. tak measured that randomising mass alone took his
    Isaac Lab -> MuJoCo jump-height gap from +29% to +6% *and* left the policy jumping
    higher, i.e. the variation acted as a regulariser rather than a tax -- while a
    policy trained at a single fixed value transferred poorly regardless of which value
    was chosen. Our own mujoco runs failed in exactly that way (fine in Isaac Sim,
    falls in MuJoCo), so both are added here.
    """

    body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.80, 1.20),
            "damping_distribution_params": (0.80, 1.20),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # 2026-09-07 (Loop 29): the torque-speed curve itself, which every other actuator
    # property was already randomised around. Drawn per RESET rather than per startup --
    # the other terms above fix a machine for the whole run, which a blind policy with an
    # observation history can still adapt to online; this one has to stay unlearnable,
    # because it is the property the take-off was found to depend on.
    #
    # See mdp.randomize_actuator_torque_speed_curve for the measurement behind it. The
    # short version: mujoco's ctrlrange is flat in speed, the knees hit 37-58 rad/s in the
    # push-off where Isaac supplies no torque, and the same policy that keeps its attitude
    # in Isaac (0.48 rad/s RMS in flight) leaves the ground with ~2 rad/s of pitch there
    # and lands inverted. speed_scale up to 2.5 puts X2 at 75 rad/s -- no effective derate
    # over the operating range -- so mujoco's behaviour is inside the training
    # distribution instead of outside it.
    # 2026-09-16: 評価のために範囲を環境変数で上書きできるようにした（既定値は据え置き）。
    # speed_scale=1.0 に固定すれば実機相当（X2=30 rad/s）、2.5 に固定すれば mujoco 相当
    # （X2=75 rad/s＝観測される動作範囲では実質デレート無し）の機体だけで測れる。
    # 既定の (0.80, 2.50) は平均 1.65 倍甘い機体の分布なので、学習ログの unaided_* は
    # 「実機相当の性能」ではない。
    actuator_torque_speed_curve = EventTerm(
        func=mdp.randomize_actuator_torque_speed_curve,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "torque_scale_range": (
                float(os.environ.get("TN_TORQUE_SCALE_MIN", 0.80)),
                float(os.environ.get("TN_TORQUE_SCALE_MAX", 1.20)),
            ),
            "speed_scale_range": (
                float(os.environ.get("TN_SPEED_SCALE_MIN", 0.80)),
                float(os.environ.get("TN_SPEED_SCALE_MAX", 2.50)),
            ),
        },
    )


@configclass
class RobotEnvCfgLongJumpBase(RobotEnvCfg):
    """Stage A: flat-ground running, extended forward-speed ceiling for a running approach."""

    commands: CommandsCfgLongJumpBase = CommandsCfgLongJumpBase()
    rewards: RewardsCfgLongJumpBase = RewardsCfgLongJumpBase()
    events: EventCfgLongJump = EventCfgLongJump()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = GO2_CORRECTED_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2026-09-16: 物理刻みを環境変数で上書きできるようにした（既定は上流の 0.005 のまま）。
        #
        # 同じ ckpt を刻みだけ変えて測ると、踏切の離陸速度の積 v_x*v_z が
        #   0.005 -> 3.851 / 0.0025 -> 4.874 / 0.002 -> 5.380 / 0.001 -> 5.444
        # で、**0.002 でほぼ収束し 0.005 は収束値より 29% 低い**（eval_vz_gap/）。
        # 踏切は 100-160 ms しかないので 0.005 では 20-30 ステップしか刻めず、
        # 接触の力積を取りこぼしている。Isaac が mujoco より跳ばない主因はこれ。
        # 制御周期 0.02 s は decimation 側で保つ。
        _sim_dt = os.environ.get("SIM_DT")
        if _sim_dt:
            self.sim.dt = float(_sim_dt)
            self.decimation = int(round(0.02 / self.sim.dt))
            self.sim.render_interval = self.decimation
            self.scene.contact_forces.update_period = self.sim.dt
            if getattr(self.scene, "height_scanner", None) is not None:
                self.scene.height_scanner.update_period = self.decimation * self.sim.dt
            print(f"[longjump] sim.dt={self.sim.dt} decimation={self.decimation} "
                  f"(step_dt={self.sim.dt * self.decimation})", flush=True)

        # 2026-09-04:真の更地(無限平面)にする。
        #
        # COBBLESTONE_ROAD_CFG はサブ地形こそ "flat" だけだが、生成器としては
        # 8m x 8m のタイルを 10x20 枚並べ、さらに幅 20m の外周ボーダーを作る。
        # このタスクの助走は 3.3 m/s で episode 長は 20 秒 -- つまり 1 エピソードで
        # 約 66 m 走るので、robot はタイル境界を何枚も跨ぎ、最後にはボーダーに
        # 突き当たる。跳躍の踏切・着地がタイル境界と重なると、学習にとっては
        # 「地形」ではなく単なるノイズになる。平地専用タスクなので生成器ごと
        # 外して無限平面にする。
        #
        # terrain_levels カリキュラムは生成器地形の行/列を前提にしているので、
        # plane と併用できない。ここで無効化する(平地のみの本タスクでは
        # そもそも昇降させる地形段階が存在しない)。
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


@configclass
class RobotPlayEnvCfgLongJumpBase(RobotEnvCfgLongJumpBase):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        # Mirror RobotPlayEnvCfg: start commands at the trained ceiling
        # instead of the curriculum's initial (-0.1, 0.1) range -- otherwise
        # play would look like a slow walk while lin_vel_cmd_levels
        # re-climbs from scratch in this freshly-constructed env (same
        # mechanism as the --resume curriculum reset, see
        # フィードバック_resume時カリキュラムリセット.md).
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
