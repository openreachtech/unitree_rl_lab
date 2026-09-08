from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class GetUpPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "go2_getup"
