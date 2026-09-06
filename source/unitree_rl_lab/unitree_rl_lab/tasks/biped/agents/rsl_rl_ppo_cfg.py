"""PPO for the unified bipedal policy: the base configuration plus a left-right mirror."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from unitree_rl_lab.tasks.biped.mdp import mirror_left_right
from unitree_rl_lab.tasks.dynamic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class BipedPPORunnerCfg(BasePPORunnerCfg):
    """Every hyper-parameter of the shared base, with each training batch augmented by its mirror.

    The robot is bilaterally symmetric and so is the task, but a policy trained without this
    invents a preferred side and keeps it. Measured on the unified policy before the change: 94% of
    its hind-stance steps on the left hind foot against 37% on the right, and the left front foot
    propping it up ten times as often as the right. The hind stance's own first run knelt on its
    left shin 8.7% of the time and its right shin 0.2%. Nothing in the reward asks for any of that.

    Mirroring the batch takes the asymmetry out of what the policy can represent rather than
    charging for it afterwards. Against an otherwise identical run it left the two sides even to
    within half a percentage point of stance-foot duty (43.6 against 43.4), cut falls from 9 in 128
    play episodes to 3, and improved velocity tracking on every axis but yaw -- planar error 0.222
    against 0.274 in the hind stance and 0.238 against 0.276 in the front.

    ``use_mirror_loss`` stays off. The augmentation alone doubles the batch and is the cheaper of
    the two mechanisms to reason about; both at once would leave two explanations for any result.

    Not applied to the single-stance tasks, whose checkpoints were trained without it. The same
    asymmetry is visible there and this would very likely help, but changing their configuration
    would leave each with a policy its own config no longer describes.
    """

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=mirror_left_right,
        )
