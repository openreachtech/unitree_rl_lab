"""Stage A for Anaguma: flat-ground running ("Anaguma-LongJump-Base-v1").

Same structure as the Go2 Stage A it inherits from -- the point of the port is that
the task survives the change of robot. What is NOT shared:

  * the robot itself (see ``unitree_rl_lab.assets.robots.anaguma``),
  * ``undesired_contacts``, because the Go2 term lists ``Head_.*`` bodies that this
    machine does not have, and an unresolvable body pattern is a hard error.

Everything else -- the reward rebalance that stopped Stage B collapsing, the forward-only
command range, the domain randomisation including the torque-speed curve -- is Go2's and
is inherited deliberately. The internal doc「AnagumaとGo2の違い」warns that applying the
unitree_rl_lab settings unchanged makes the robot "unable to walk at all", so the first
run is a test of exactly that claim; the things it names (solver iteration counts,
base_link not being at the CoM) are handled in the robot config and in the reward set
respectively, and anything left over should show up as a failure to walk.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.anaguma import ANAGUMA_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.longjump_base_env_cfg import (
    RewardsCfgLongJumpBase,
    RobotEnvCfgLongJumpBase,
)


@configclass
class RewardsCfgAnagumaLongJumpBase(RewardsCfgLongJumpBase):
    """Go2's Stage A rewards, with the head removed from the contact penalty."""

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]
            ),
        },
    )


@configclass
class RobotEnvCfgAnagumaLongJumpBase(RobotEnvCfgLongJumpBase):
    """Stage A on Anaguma: flat ground, forward-only, extended speed ceiling."""

    rewards: RewardsCfgAnagumaLongJumpBase = RewardsCfgAnagumaLongJumpBase()

    def __post_init__(self):
        super().__post_init__()
        # Replaces the Go2 articulation that the parent installs (its corrected actuator
        # model is Go2's knee, and does not describe this machine).
        self.scene.robot = ANAGUMA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class RobotPlayEnvCfgAnagumaLongJumpBase(RobotEnvCfgAnagumaLongJumpBase):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        # Start at the trained ceiling rather than letting lin_vel_cmd_levels re-climb
        # from scratch in a freshly built env (フィードバック_resume時カリキュラムリセット.md).
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
