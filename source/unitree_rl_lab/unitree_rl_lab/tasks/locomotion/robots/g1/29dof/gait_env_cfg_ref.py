import math
import copy
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.commands import GaitReferenceCommandCfg
from .velocity_env_cfg import RewardsCfg, RobotEnvCfg, RobotPlayEnvCfg
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.terrains import TerrainGeneratorCfg
from .velocity_env_cfg import ObservationsCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCasterCfg, patterns

ROUGH_TERRAINS__NOSTAIR_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,          # 難易度の段階数
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.2), platform_width=2.0, border_width=0.25
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.2), platform_width=2.0, border_width=0.25
        ),
    },
)
ROUGH_TERRAINS_WITH_STAIR_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=10,
    horizontal_scale=0.15,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.2), platform_width=2.0, border_width=0.25
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.2), platform_width=2.0, border_width=0.25
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.03, 0.17),   # difficulty で線形補間される
            step_width=0.30,                  # 踏面
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.03, 0.17),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

# 左右のペアを揃えて使う（フォークの実装が左右対応を要求する）
ARMS = SceneEntityCfg("robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"])
KNEES = SceneEntityCfg("robot", joint_names=["left_knee_joint", "right_knee_joint"])
FEET  = SceneEntityCfg("contact_forces",

                       body_names=["left_ankle_roll_link", "right_ankle_roll_link"])
WAIST_PITCH = SceneEntityCfg("robot", joint_names=["waist_pitch_joint"])


@configclass
class GaitRewardsCfg(RewardsCfg):
    """既存の報酬に、歩容ごとに効く報酬を足す。"""

    # ===== フォークに合わせて追加 =====
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.2,
        },
    )

    dof_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    # --- STAND (id 0)
    s_knee = RewTerm(func=mdp.stance_knee_extension, weight=0.5,
                     params={"knee_cfg": KNEES, "foot_sensor_cfg": FEET,
                             "target": 0.12, "scale": 5.0, "target_gait_id": 0})
    s_arm_dev = RewTerm(func=mdp.gait_joint_deviation_l1, weight=-0.1,
                        params={"asset_cfg": ARMS, "target_gait_id": 0})

    # --- WALK (id 1)
    w_knee = RewTerm(func=mdp.stance_knee_extension, weight=0.2,
                     params={"knee_cfg": KNEES, "foot_sensor_cfg": FEET,
                             "target": 0.18, "scale": 8.0, "target_gait_id": 1})
    w_arm_dev = RewTerm(func=mdp.gait_joint_deviation_l1, weight=-0.05,
                        params={"asset_cfg": ARMS, "target_gait_id": 1})

    # --- WALK→STAND (id 2)
    w2s_knee = RewTerm(func=mdp.stance_knee_extension, weight=0.2,
                       params={"knee_cfg": KNEES, "foot_sensor_cfg": FEET,
                               "target": 0.14, "scale": 5.0, "target_gait_id": 2})
    w2s_arm_dev = RewTerm(func=mdp.gait_joint_deviation_l1, weight=-0.08,
                          params={"asset_cfg": ARMS, "target_gait_id": 2})
    # --- RUN (id 3) / RUN→WALK (id 4)
    r_arm_dev = RewTerm(func=mdp.gait_joint_deviation_l1, weight=-0.02,
                        params={"asset_cfg": ARMS, "target_gait_id": 3})
    r2w_arm_dev = RewTerm(func=mdp.gait_joint_deviation_l1, weight=-0.03,
                          params={"asset_cfg": ARMS, "target_gait_id": 4})
                              # 走行・走行→歩行で、上体が骨盤に対して倒れるのを罰する
    # 全 gait で、上体が骨盤に対して倒れるのを罰する
    waist_pitch_dev = RewTerm(
        func=mdp.gait_joint_pos_l2,
        weight=-2.0,
        params={"asset_cfg": WAIST_PITCH, "target_gait_id": None},
    )
        # 腰ロール: 歩行に機能的な役割がないので強めに拘束
    waist_roll_dev = RewTerm(
        func=mdp.gait_joint_pos_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint"]),
            "target_gait_id": None,
        },
    )
    # 腰ヨー: 脚の反動を上体で打ち消す機能があるので緩めに
    waist_yaw_dev = RewTerm(
        func=mdp.gait_joint_pos_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"]),
            "target_gait_id": None,
        },
    )


@configclass
class GaitEnvCfg(RobotEnvCfg):
    rewards: GaitRewardsCfg = GaitRewardsCfg()

    # gait_env.py が参照するフィールド（これが無いと AttributeError）
    leg_length: float = 0.6
    curriculum_phase: int = 3
    phase_command_curriculum: dict = {}


    def __post_init__(self):
        super().__post_init__()

        # --- まず平地から
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None      # 生成地形が無いので必ず外す
                # --- 平地からラフ地形へ
        #self.scene.terrain.terrain_type = "generator"
        #self.scene.terrain.terrain_generator = ROUGH_TERRAINS__NOSTAIR_CFG
        #lJself.scene.terrain.max_init_terrain_level = 5
        #self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        # --- 指令カリキュラムは phase 方式に一本化（二重に動かさない）
        self.curriculum.lin_vel_cmd_levels = None

        # --- ドメインランダム化（フォークの値）
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.9)
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.push_robot.interval_range_s = (10.0, 15.0)

        # --- 段階別の指令レンジ
        self.phase_command_curriculum = {
            "1": {"resampling_time_range": (10.0, 10.0), "rel_standing_envs": 0.02,
                  "lin_vel_x": (0.0, 1.0), "lin_vel_y": (-0.5, 0.5),
                  "ang_vel_z": (-1.0, 1.0), "heading": (-math.pi, math.pi)},
            "2": {"resampling_time_range": (5.0, 10.0), "rel_standing_envs": 0.3,
                  "lin_vel_x": (0.0, 1.0), "lin_vel_y": (-0.5, 0.5),
                  "ang_vel_z": (-1.0, 1.0), "heading": (-math.pi, math.pi)},
            "3": {"resampling_time_range": (5.0, 10.0), "rel_standing_envs": 0.1,
                  "lin_vel_x": (0.0, 2.5), "lin_vel_y": (-0.5, 0.5),
                  "ang_vel_z": (-1.0, 1.0), "heading": (-math.pi, math.pi)},
        }

        # --- 歩容 one-hot を観測に入れる場合はここを有効にする
        #     入れると実機側でも歩容を計算する必要が出る（要判断）
        # self.observations.policy.gait_onehot = ObsTerm(func=mdp.gait_onehot)
        # self.observations.critic.gait_onehot = ObsTerm(func=mdp.gait_onehot)
                # ===== 報酬をフォーク（IsaacLab）の設定に合わせる =====
        # 動きを抑える方向のペナルティを緩める
        self.rewards.base_linear_velocity.weight = -0.2      # 鉛直速度 (-2.0)
        self.rewards.action_rate.weight = -0.005             # 行動変化率 (-0.05)
        self.rewards.joint_acc.weight = -1.0e-7              # 関節加速度 (-2.5e-7)
        #self.rewards.flat_orientation_l2.weight = -1.0       # 姿勢の水平維持 (-5.0)
                # 姿勢: 前傾は 20 度まで無罰、後傾とロールは罰する
        self.rewards.flat_orientation_l2.func = mdp.gait_orientation_asym_l2
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.flat_orientation_l2.params = {
            "free_pitch_forward": 0.35,
            "roll_scale": 1.0,
            "pitch_back_scale": 1.5,      # 後傾は前傾より 1.5 倍嫌う
            "target_gait_id": None,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.rewards.joint_deviation_legs.weight = -0.3      # 股関節の固定 (-1.0)
        self.rewards.feet_slide.weight = -0.1                # 足の滑り (-0.2)
        self.rewards.track_ang_vel_z.weight = 1.0            # 旋回追従 (0.5)

        # 足首のみに限定して緩める
        self.rewards.dof_pos_limits.weight = -1.0            # (-5.0)
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
        )

        # フォークに無い項目を無効化
        self.rewards.base_height = None                      # 胴体高さの維持 (-10.0)
        self.rewards.joint_deviation_waists = None           # 腰の固定 (-1.0)
        #self.rewards.gait = None                             # 周期 0.8 秒の接地パターン
        #self.rewards.feet_clearance = None                   # 足の持ち上げ (feet_air_time で代替)
        self.rewards.gait.weight = 1.0          # 0.5 → 1.0
        self.rewards.feet_clearance.weight = 1.5  # 1.0 → 1.5
        self.rewards.feet_clearance.params["target_height"] = 0.10
        self.rewards.feet_clearance.params["std"] = 0.03   # 0.05 → 0.03 で勾配を立てる
          # 腕の二重拘束を解消（歩容別の *_arm_dev だけに任せる）
        self.rewards.joint_deviation_arms.weight = -0.02   # -0.1 → -0.02

        self.rewards.undesired_contacts = None
        self.rewards.joint_vel = None
        self.rewards.energy = None                           # dof_torques で代替
        self.rewards.alive = None                            # termination_penalty で代替
        self.scene.height_scanner = None


@configclass
class GaitPlayEnvCfg(GaitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5

        #self.viewer.origin_type = "world"
        #self.viewer.eye = (15.0, 15.0, 12.0)
        #self.viewer.lookat = (0.0, 0.0, 0.0)
        #self.viewer.resolution = (1280, 720)
         # --- ロボット追従カメラ（eye/lookat はロボット原点からの相対オフセット）
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (0.0, -3.0, -0.2)      # 真横、やや低め
        self.viewer.lookat = (0.0, 0.0, -0.4)    # 脚のあたりを見る
        self.viewer.resolution = (1280, 720)
@configclass
class StairsObservationsCfg(ObservationsCfg):
    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        # velocity_env_cfg でコメントアウトされている項目を、継承して有効化
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class GaitStairsEnvCfg(GaitEnvCfg):
    #observations: StairsObservationsCfg = StairsObservationsCfg()

    def __post_init__(self):
        super().__post_init__()      # ここで plane 化 + height_scanner=None がかかる
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        # --- 地形を階段入りに ---
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_WITH_STAIR_CFG
        self.scene.terrain.max_init_terrain_level = 0

                # ===== 階段用の報酬調整 =====

        # (1) クリアランスを地形相対に
        self.rewards.feet_clearance.func = mdp.foot_clearance_reward_relative
        self.rewards.feet_clearance.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[".*_ankle_roll_link"]
        )
        self.rewards.feet_clearance.params["target_height"] = 0.14   # 0.10 → 0.14
        self.rewards.feet_clearance.params["std"] = 0.06             # 0.03 → 0.06

        # (2) 固定周期の拘束を緩める
        self.rewards.gait.weight = 0.3                               # 1.0 → 0.3

        # (3) 鉛直速度ペナルティを緩める（登坂を罰しないように）
        self.rewards.base_linear_velocity.weight = -0.05             # -0.2 → -0.05

        # (4) 支持脚の膝伸展を許容側に（scale を下げて減衰を緩く）
        self.rewards.w_knee.params["scale"] = 3.0                    # 8.0 → 3.0
        self.rewards.w2s_knee.params["scale"] = 3.0                  # 5.0 → 3.0
                # (5) 遊脚の衝突回避 — TO でいう「遊脚が段の垂直面に当たらない」拘束。
        #     feet_clearance は「段の上面より高く上げろ」としか言っておらず、
        #     「垂直面にぶつかるな」に相当する項が無かった。両方あって初めて
        #     遊脚の通り道が上下から挟まれる。
        self.rewards.feet_stumble = RewTerm(
            func=mdp.feet_stumble,
            weight=-2.0,
            params={"sensor_cfg": FEET, "ratio": 4.0, "min_force": 10.0},
        )

        # (6) 脚以外の接地を復活。膝〜足首のリンク（G1 では *_knee_link が脛）が
        #     段鼻に当たる階段の典型的な失敗が、これまで終了まで無罰だった。
        #     リンク名は `print(env.scene["robot"].body_names)` で要確認。
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*_knee_link"]
                ),
            },
        )
        # --- 終了条件も地形相対に ---
        self.terminations.base_height.func = mdp.root_height_below_minimum_relative
        self.terminations.base_height.params["feet_cfg"] = SceneEntityCfg(
            "robot", body_names=[".*_ankle_roll_link"]
        )
        # minimum_height は現行値から足の厚み分（約0.03）引いた値に
        self.terminations.base_height.params["minimum_height"] = 0.1 - 0.03

        # --- height_scanner を復活（最終行で None にされている）---
        #self.scene.height_scanner = RayCasterCfg(
        #    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        #    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        #J    ray_alignment="yaw",
        #    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        #    debug_vis=False,
        #    mesh_prim_paths=["/World/ground"],
        #)
        # 基底の __post_init__ は走り終わっているので手動で
        #self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class GaitStairsPlayEnvCfg(GaitStairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5

        # --- 撮影用地形: 階段のみ・本番スケール固定 ---
        terrain = copy.deepcopy(self.scene.terrain.terrain_generator)
        terrain.num_rows = 5
        terrain.num_cols = 4
        terrain.curriculum = False
        terrain.difficulty_range = (0.7, 1.0)        # 蹴上 0.13〜0.17m
        terrain.sub_terrains = {
            "stairs_up": terrain.sub_terrains["stairs_up"],
            "stairs_down": terrain.sub_terrains["stairs_down"],
        }
        terrain.sub_terrains["stairs_up"].proportion = 0.5
        terrain.sub_terrains["stairs_down"].proportion = 0.5
        self.scene.terrain.terrain_generator = terrain
        self.scene.terrain.max_init_terrain_level = terrain.num_rows - 1

        # --- 指令を前進固定（放っておくと旋回して階段から離れる）---
        cmd = self.commands.base_velocity
        cmd.resampling_time_range = (1000.0, 1000.0)
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.8, 0.8)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)
        # gait_env が phase 辞書から上書きする経路にも同じ値を入れておく
        self.phase_command_curriculum[str(self.curriculum_phase)] = {
            "resampling_time_range": (1000.0, 1000.0),
            "rel_standing_envs": 0.0,
            "lin_vel_x": (0.8, 0.8),
            "lin_vel_y": (0.0, 0.0),
            "ang_vel_z": (0.0, 0.0),
            "heading": (0.0, 0.0),
        }

        # --- ロボット追従カメラ ---
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0 # 階段の中央あたりに追従h
        self.viewer.eye = (0.0, -3.0, 0.5)
        self.viewer.lookat = (0.0, 0.0, -0.2)
        self.viewer.resolution = (1280, 720)


        #self.scene.num_envs = 32
        #self.scene.env_spacing = 2.5
        #self.scene.terrain.terrain_generator.num_rows = 2
        #self.scene.terrain.terrain_generator.num_cols = 10

        #self.viewer.origin_type = "asset_root"
        #self.viewer.asset_name = "robot"
        #self.viewer.env_index = 0
        #self.viewer.eye = (0.0, -3.0, -0.2)
        #self.viewer.lookat = (0.0, 0.0, -0.4)
        #self.viewer.resolution = (1280, 720)

  # ======================================================================
# 条件B: 地形条件付きの粗い基準軌道で誘導する設定
#
# 上の GaitEnvCfg / GaitStairsEnvCfg（条件A）は一切変更しないので、
# 条件Aはいつでも再現できる。
#
#   条件A の歩容整形: feet_gait (period 0.8s 固定 / threshold 0.55) + feet_air_time
#   条件B の歩容整形: ref_contact / ref_com_vz / ref_angmom / ref_foot / ref_clearance
#
# 項          適用区間                          役割
# ---------------------------------------------------------------------
# ref_contact 全区間                            左右足の接触系列
# ref_com_vz  走行=滞空期 / 階段=単脚支持期      重心の上下移動
# ref_angmom  走行=滞空期 / 階段=単脚支持期      全身の回転量
# ref_foot    着地の瞬間                        地形から求めた着地点
# ref_clearance 遊脚期                          段との衝突を未然に防ぐ
# load_torque 全区間                            モーター負荷
# feet_stumble / undesired_contacts 全区間       衝突の事後罰（条件A と共通）
#
# 追従・姿勢・平滑化・安全の各項は両条件で同一。
# ======================================================================


@configclass
class GaitRefRewardsCfg(GaitRewardsCfg):
    """条件B。ここでは項を「足す」だけ。

    無効化は EnvCfg 側の __post_init__ で行う。GaitEnvCfg.__post_init__ が
    self.rewards.gait.weight を触るため、クラス定義側で None にすると
    super().__post_init__() が None.weight で落ちる。

    足・関節にわたる量はすべて平均なので、本数や関節数に依存せず
    weight を調整できる。
    """

    ref_contact = RewTerm(
        func=mdp.ref_contact_match,
        weight=2.0,
        params={"sensor_cfg": FEET, "command_name": "gait_ref"},
    )
    ref_com_vz = RewTerm(
        func=mdp.ref_com_vz,
        weight=0.5,
        params={"std": 0.15, "command_name": "gait_ref"},
    )
    ref_angmom = RewTerm(
        func=mdp.ref_angular_momentum,
        weight=0.3,
        params={"std": 2.0, "yaw_std": 0.30,"command_name": "gait_ref"},
    )
    # --- 地形条件付きの中核 2 項 ---
    ref_foot = RewTerm(
        func=mdp.ref_foot_placement,
        weight=2.0,   # 着地の瞬間しか立たない疎な信号なので、他項より高め
        params={
            "std": 0.10,
            "command_name": "gait_ref",
            # asset_cfg / sensor_cfg は params に入れないと manager が resolve せず、
            # body_ids が slice(None) のまま全リンクを指してしまう
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "sensor_cfg": FEET,
        },
    )
    ref_clearance = RewTerm(
        func=mdp.ref_swing_clearance,
        weight=-10.0,
        params={
            "margin": 0.03,
            "command_name": "gait_ref",
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "sensor_cfg": FEET,
        },
    )

    load_torque = RewTerm(
        func=mdp.load_torque_normalized,
        weight=-0.5,   # 和 -> 平均にしたぶん、以前の -0.02 から換算
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


def _apply_gait_reference(cfg, use_terrain_scan: bool = True) -> None:
    """条件B 共通の差し替え。平地・階段のどちらからも呼ぶ。

    use_terrain_scan=False にすると地形スキャナを作らず、地形 z=0 / 勾配 0 の
    フォールバックで動く。平地タスクではこれが厳密に正しい答えなので、
    VRAM を無駄に使わないために既定で切る。
    地形まわりのコードを平地で通しておきたいときだけ True にする。
    """

    # --- 地形を報酬だけが見る（観測には入れない = ブラインド条件のまま）---
    # ref_foot の踏面スナップと ref_clearance の安全高さに必要。
    #
    # upstream 標準の height_scanner (1.6x1.0 / 0.1m) ではなく、解像度 0.05m の
    # 足元用スキャンを別に置く。0.1m 格子だと踏面 0.30m に対して着地点の
    # 不確かさが ±0.05m 出て、足裏長 0.20m が踏面に残す余裕（片側 0.05m）を
    # 食い潰すため。範囲は 1.4x0.5m = 29x11 = 319 レイで、
    # 着地点（前方 ~0.35m）と勾配プローブ（前方 0.6m）の両方が入る。
    #
    # 将来ポリシーへ地形を渡すときは、この報酬用とは別に upstream 標準の
    # height_scanner を足す想定（用途が違うので解像度も別でよい）。
    if use_terrain_scan:
        cfg.scene.foothold_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.4, 0.5]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        # 基底の __post_init__ は走り終わっているので手動で
        cfg.scene.foothold_scanner.update_period = cfg.decimation * cfg.sim.dt

    # --- 基準軌道コマンドを追加 ---
    cfg.commands.gait_ref = GaitReferenceCommandCfg(
        velocity_command_name="base_velocity",
        leg_length=cfg.leg_length,
        stance_width=0.20,
        terrain_sensor_name="foothold_scanner" if use_terrain_scan else None,
        angmom_tolerance=0.2,
        debug_vis=True,   # 最初は必ず True。基準を目で見てから weight を入れる
    )
    # --- 観測の位相を、基準軌道が使っている位相へ差し替える ---
    # ここが最重要。upstream の gait_phase は period 0.8s 固定の時計で、
    # 条件A の feet_gait はこれと同じ時計で採点していたから学習できていた。
    # 基準軌道の位相は速度依存かつ env ごとにランダム初期化なので、
    # 観測を差し替えないと ref_* は「観測できない位相で採点する」ことになり、
    # ポリシーには学習不能なノイズにしかならない。
    # 次元は 2 のままなので観測ベクトルの形は条件A と同じに保たれる。
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.gait_phase = ObsTerm(
            func=mdp.gait_reference_phase, params={"command_name": "gait_ref"}
        )
    # --- 基準が代替する手作りの歩容整形を外す ---
    cfg.rewards.gait = None            # period 0.8s / threshold 0.55 固定。滞空と両立しない
    cfg.rewards.feet_air_time = None   # 接地時間は基準スケジュールが決める

    # --- 腕の角運動量を使えるように shoulder_pitch を拘束から外す ---
    # 滞空中の h 保存を守るには腕が振れる必要があり、shoulder_pitch を
    # 既定姿勢へ引き戻したままだと ref_angmom が原理的に満たせない。
    #
    # 各項に必ず「別インスタンス」を渡すこと。SceneEntityCfg.resolve() は
    # 対象をその場で書き換えて joint_ids を埋めるが joint_names は正規表現の
    # まま残すので、同じインスタンスを使い回すと 2 つ目以降の resolve で
    # 「joint_names(正規表現) と joint_ids(解決済み) が不整合」と判定される。
    def _arms_no_pitch() -> SceneEntityCfg:
        return SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*",
            ],
        )

    cfg.rewards.joint_deviation_arms.params["asset_cfg"] = _arms_no_pitch()
    for name in ("s_arm_dev", "w_arm_dev", "w2s_arm_dev", "r_arm_dev", "r2w_arm_dev"):
        getattr(cfg.rewards, name).params["asset_cfg"] = _arms_no_pitch()

     # --- 片手振り対策: 肩ピッチの反対称性だけを課す
    cfg.rewards.arm_antisym = RewTerm(
        func=mdp.arm_swing_antisymmetry,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"],
            )
        },
    )

@configclass
class GaitRefEnvCfg(GaitEnvCfg):
    """条件B / 平地・走行。

    平地では地形の答えが自明（z=0 / 勾配 0）なので、地形スキャナは作らない。
    ref_foot は Raibert の名目位置をそのまま目標にし、ref_clearance は
    z_safe = margin の弱い下限として働く。地形条件付きの部分が実際に
    仕事をするのは階段側。
    """

    rewards: GaitRefRewardsCfg = GaitRefRewardsCfg()
    curriculum_phase: int = 3   # A と同じ手順で 1 -> 2 -> 3 と積み上げる

    def __post_init__(self):
        super().__post_init__()
        _apply_gait_reference(self, use_terrain_scan=False)


@configclass
class GaitRefPlayEnvCfg(GaitRefEnvCfg):
    curriculum_phase: int = 3   # 確認したい歩容に合わせる（1 のままだと走行が出ない）

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (0.0, -3.0, -0.2)
        self.viewer.lookat = (0.0, 0.0, -0.4)
        self.viewer.resolution = (1280, 720)


@configclass
class GaitRefStairsEnvCfg(GaitStairsEnvCfg):
    """条件B / 階段。

    平地との差は地形と衝突罰だけで、報酬の構成は同じ。
    ref_com_vz / ref_angmom は phase_mask が歩容で切り替わるので、
    階段（WALK 側 = 飛翔期なし）では単脚支持期に適用される。
    ref_foot / ref_clearance は foothold_scanner 経由で段の形状を見る。
    """

    rewards: GaitRefRewardsCfg = GaitRefRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_gait_reference(self)


@configclass
class GaitRefStairsPlayEnvCfg(GaitRefStairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5

        terrain = copy.deepcopy(self.scene.terrain.terrain_generator)
        terrain.num_rows = 5
        terrain.num_cols = 4
        terrain.curriculum = False
        terrain.difficulty_range = (0.7, 1.0)
        terrain.sub_terrains = {
            "stairs_up": terrain.sub_terrains["stairs_up"],
            "stairs_down": terrain.sub_terrains["stairs_down"],
        }
        terrain.sub_terrains["stairs_up"].proportion = 0.5
        terrain.sub_terrains["stairs_down"].proportion = 0.5
        self.scene.terrain.terrain_generator = terrain
        self.scene.terrain.max_init_terrain_level = terrain.num_rows - 1

        cmd = self.commands.base_velocity
        cmd.resampling_time_range = (1000.0, 1000.0)
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.8, 0.8)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)
        self.phase_command_curriculum[str(self.curriculum_phase)] = {
            "resampling_time_range": (1000.0, 1000.0),
            "rel_standing_envs": 0.0,
            "lin_vel_x": (0.8, 0.8),
            "lin_vel_y": (0.0, 0.0),
            "ang_vel_z": (0.0, 0.0),
            "heading": (0.0, 0.0),
        }

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (0.0, -3.0, 0.5)
        self.viewer.lookat = (0.0, 0.0, -0.2)
        self.viewer.resolution = (1280, 720) 