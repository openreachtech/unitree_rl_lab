from isaaclab.utils import configclass

from unitree_rl_lab.assets.models.biped_actor import BipedActorCritic
from unitree_rl_lab.assets.models.modules.biped_ppo import BipedPPO
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

# Number of stacked frames (current step + past) fed to BipedEstimator. Kept here
# (rather than in the env cfg) so the obs-group history_length used by
# ``robots/go2/biped_env_cfg.py`` and the dims declared below can never drift apart.
PROPRIO_HISTORY_LENGTH = 4

# ang_vel(3) + projected_gravity(3) + velocity_commands(3) + gait_mode(3)
# + joint_pos_rel(12) + joint_vel_rel(12) + last_action(12) -- current single step.
# TumblerNet reference's raw proprioceptive observation is 45-dim (no gait_mode);
# this task now also carries a one-hot gait-mode observation (permanently pinned
# to hind-biped for now, see ``robots/go2/biped_env_cfg.py``), adding 3 dims.
PROPRIO_TERM_DIM = 3 + 3 + 3 + 3 + 12 + 12 + 12

# BipedPolicy "policy" obs group layout: [current proprio | stacked history].
PROPRIO_OBS_DIM = PROPRIO_TERM_DIM
HISTORY_OBS_DIM = PROPRIO_TERM_DIM * PROPRIO_HISTORY_LENGTH

# BipedCritic "critic" obs group layout: current proprio + true lin_vel(3)
# + true com_cop(3) + joint_effort(12).
CRITIC_OBS_DIM = PROPRIO_TERM_DIM + 3 + 3 + 12


@configclass
class BipedPPORunnerCfg(BasePPORunnerCfg):
    """Use the custom biped actor-critic (+ jointly-trained estimator) with rsl-rl PPO."""

    def __post_init__(self):
        super().__post_init__()
        # Ensure class symbols are imported in this module for config serialization/debug.
        _ = BipedActorCritic
        _ = BipedPPO
        self.policy.class_name = "BipedActorCritic"
        self.policy.proprio_obs_dim = PROPRIO_OBS_DIM
        self.policy.history_obs_dim = HISTORY_OBS_DIM
        self.policy.critic_obs_dim = CRITIC_OBS_DIM
        self.algorithm.class_name = "BipedPPO"
        self.algorithm.aux_loss_coef = 0.5
