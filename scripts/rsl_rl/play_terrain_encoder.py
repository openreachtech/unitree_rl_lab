# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch the terrain encoder's estimate next to what the sensor actually gave it.

Two point clouds, no ground truth:

    green   a cell the fan measured this step
    red     a cell holding an older reading -- what the sensor cannot see right now
    blue    the encoder's reconstruction

``--hide_observed`` drops the green and red and leaves the estimate on its own, which is
the view to use for judging the reconstructed surface as a surface: with all three clouds
up, the sensor's points sit close enough to the estimate to obscure it.

Blue covers the whole grid, red and green only the 388 cells the observation keeps, so
the body footprint comes out blue-only. Nothing has ever been measured there -- the fan
fires from one point 52 cm up and the near field is a structural blind cone -- so
whatever the blue does under the robot came out of the recurrent state and nowhere else.
That patch is the clearest thing to watch.

The true terrain is deliberately not drawn. It would sit underneath both clouds and make
the comparison that matters harder to read; ``RMSE/body`` in the training log is the
place to get the number.

    python scripts/rsl_rl/play_terrain_encoder.py \
        --policy_checkpoint logs/rsl_rl/go2_blind_gru_phase2/<run>/model_6497.pt \
        --encoder_checkpoint logs/terrain_encoder/<run>/encoder_1000.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained terrain encoder.")
parser.add_argument("--task", type=str, default="Go2-Terrain-Encoder-Phase2-Play")
parser.add_argument("--policy_checkpoint", type=str, required=True, help="Frozen walking policy.")
parser.add_argument("--encoder_checkpoint", type=str, required=True, help="Trained terrain encoder.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument(
    "--estimate_cells",
    choices=("all", "observed"),
    default="all",
    help="Draw the estimate over the whole grid, or only the cells the observation keeps.",
)
parser.add_argument(
    "--hide_observed",
    action="store_true",
    default=False,
    help="Turn off the green/red sensor markers and leave only the blue estimate.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.assets.models.modules.runners import UnitreeOnPolicyRunner
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import (
    GO2_HEIGHT_SCAN_OFFSET,
    HEIGHT_SCAN_SIZE,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_terrain_encoder import (
    build_terrain_encoder,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def cell_offsets(grid_shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    """``(H*W, 2)`` cell centres in the base frame, in the grid's flatten order.

    Mirrors the layout ``_height_scan_indices`` produces -- x over rows, y over
    columns, ``idx = ix * num_y + iy`` -- so element ``i`` here is the same cell as
    element ``i`` of the encoder's flattened output.
    """
    num_x, num_y = grid_shape
    x = torch.linspace(-HEIGHT_SCAN_SIZE[0] / 2, HEIGHT_SCAN_SIZE[0] / 2, num_x, device=device)
    y = torch.linspace(-HEIGHT_SCAN_SIZE[1] / 2, HEIGHT_SCAN_SIZE[1] / 2, num_y, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)


class EstimateMarkers:
    """Blue spheres at the encoder's reconstructed heights."""

    def __init__(self, grid_shape: tuple[int, int], device: torch.device):
        self.offsets = cell_offsets(grid_shape, device)
        self.markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Go2TerrainEstimate",
                markers={
                    "estimate": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 1.0)),
                    )
                },
            )
        )

    def draw(self, robot, mean: torch.Tensor, keep: torch.Tensor | None) -> None:
        """``mean`` is ``(N, H, W)`` in metres, on the same convention as the input.

        A cell's height is ``base_z - value - offset``, the inverse of how the scan
        produced the value in the first place, so the marker lands on the surface the
        encoder believes is there.
        """
        flat = mean.flatten(start_dim=1)
        offsets = self.offsets
        if keep is not None:
            flat = flat.index_select(1, keep)
            offsets = offsets.index_select(0, keep)

        root_pos = robot.data.root_pos_w
        quat = robot.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        ).unsqueeze(-1)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        cell_x, cell_y = offsets[:, 0].unsqueeze(0), offsets[:, 1].unsqueeze(0)

        positions = torch.stack(
            [
                root_pos[:, 0:1] + cos_y * cell_x - sin_y * cell_y,
                root_pos[:, 1:2] + sin_y * cell_x + cos_y * cell_y,
                root_pos[:, 2:3] - flat - GO2_HEIGHT_SCAN_OFFSET,
            ],
            dim=-1,
        ).reshape(-1, 3)
        finite = torch.isfinite(positions).all(dim=-1)
        if bool(finite.any()):
            self.markers.visualize(translations=positions[finite])


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.hide_observed:
        # The green/red cloud is drawn by the LiDAR observation term itself, not by this
        # script, so switching it off means switching off that term's debug_vis. The term
        # still runs -- the encoder is reading it -- it just stops drawing.
        env_cfg.observations.lidar_map.height_scan.params["debug_vis"] = False

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    device = torch.device(agent_cfg.device)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = UnitreeOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.policy_checkpoint)
    policy = runner.get_inference_policy(device=device)
    runner.alg.policy.eval()
    print(f"[INFO] Frozen policy from {args_cli.policy_checkpoint}", flush=True)

    # Width comes from the checkpoint, not from a default, so a run trained at some
    # other size still loads.
    state = torch.load(args_cli.encoder_checkpoint, map_location=device)
    encoder = build_terrain_encoder(
        device,
        extero_channels=state.get("extero_channels", 16),
        hidden_channels=state.get("hidden_channels", 16),
        arch=state.get("arch", "convgru"),  # pre-dates the field: those runs were all ConvGRU
        belief_latent=state.get("belief_latent", 96),
    )
    encoder.load_state_dict(state["encoder"])
    encoder.eval()
    print(
        f"[INFO] Encoder from {args_cli.encoder_checkpoint}"
        f" (trained to iteration {state.get('iteration', '?')})",
        flush=True,
    )

    markers = EstimateMarkers(encoder.grid_shape, device)
    keep = None if args_cli.estimate_cells == "all" else encoder.keep_index
    robot = env.unwrapped.scene["robot"]

    hidden = encoder.init_hidden(env.num_envs, device)
    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            height = encoder.scatter_observation(obs["lidar_map"])
            mean, _, hidden = encoder(height, obs["policy"], hidden)
            markers.draw(robot, mean, keep)
            obs, _, done, _ = env.step(policy(obs))
            # Same rule the trainer uses: a reset means the remembered terrain belongs
            # to a patch of ground the robot is no longer standing on.
            hidden = encoder.mask_hidden(hidden, done.bool())

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
