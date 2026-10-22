"""Export a checkpoint for deployment: ``exported/policy.onnx`` + ``params/deploy.yaml`` beside it.

``train.py`` writes ``params/deploy.yaml`` and ``play.py`` writes the ONNX, so a checkpoint that was
assembled rather than trained (``build_recovery_moe_checkpoint.py``) has neither. This writes both
from the task's training environment -- the one whose command ranges, gains and observation terms
the deploy side has to reproduce -- without stepping it.

    python scripts/rsl_rl/export_policy.py --task Unitree-Go2-GetUp-v0 \
        --checkpoint logs/rsl_rl/go2_getup/<run>/model_0.pt

The resulting run directory is what ``deploy/robots/go2/config/config.yaml``'s ``policy_dir`` points at.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import os  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg  # noqa: E402


class ClampedActor(torch.nn.Module):
    """The actor followed by the runner's ``clip_actions``, so the exported network clamps its own output.

    ``RslRlVecEnvWrapper`` clips actions before ``env.step``, outside the network: training only ever
    saw clipped actions, and the clipped value is also what comes back as ``last_action``. Exported
    without it, a deploy side that clips more loosely (go2_ctrl and anaguma_main both use +-100) runs a
    different policy -- one whose large outputs feed back through ``last_action`` and run away, which
    is how the walk / fall / get-up mixture ended up frozen on its side in MuJoCo with actions near 300.
    """

    def __init__(self, actor: torch.nn.Module, clip: float):
        super().__init__()
        self.actor = actor
        self.clip = float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.actor(x), -self.clip, self.clip)

    def __getitem__(self, index: int):
        # the ONNX exporter reads `actor[0].in_features` for its dummy input
        return self.actor[index]


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    checkpoint = os.path.abspath(args_cli.checkpoint)
    run_dir = os.path.dirname(checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    export_deploy_cfg(env.unwrapped, run_dir)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)

    policy_nn = runner.alg.policy
    if agent_cfg.clip_actions is not None:
        policy_nn.actor = ClampedActor(policy_nn.actor, agent_cfg.clip_actions)
        print(f"[INFO] clip_actions={agent_cfg.clip_actions} is built into the exported network")
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_dir = os.path.join(run_dir, "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    print(f"[INFO] wrote {os.path.join(run_dir, 'params', 'deploy.yaml')} and {export_dir}/policy.onnx")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
