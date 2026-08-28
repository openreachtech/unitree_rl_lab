"""Replay a mimic motion npz and record it to an mp4 (no display required).

Usage:
    python replay_npz_video.py -f path_to_motion.npz --robot go2 --output out.mp4
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a mimic motion npz and record it to a video.")
parser.add_argument("--file", "-f", type=str, required=True)
parser.add_argument("--robot", type=str, default="g1", choices=["g1", "go2"], help="Which robot's motion to replay.")
parser.add_argument("--output", "-o", type=str, required=True, help="Output mp4 path.")
parser.add_argument("--fps", type=int, default=50, help="Output video fps (should match the npz fps).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG, UNITREE_GO2_CFG
from unitree_rl_lab.tasks.mimic.mdp import MotionLoader

ROBOT_CFG = UNITREE_GO2_CFG if args_cli.robot == "go2" else UNITREE_G1_29DOF_CFG
# follow-camera offset from the robot root, scaled to each robot's body size
CAMERA_OFFSETS = {
    "g1": ((2.8, 2.8, 1.0), (0.0, 0.0, 0.0)),
    "go2": ((1.6, 1.6, 0.7), (0.0, 0.0, -0.1)),
}
EYE_OFFSET, LOOKAT_OFFSET = CAMERA_OFFSETS[args_cli.robot]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    import omni.replicator.core as rep

    # Extract scene entities
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion = MotionLoader(
        args_cli.file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )

    resolution = (1280, 720)
    render_product = rep.create.render_product("/OmniverseKit_Persp", resolution)
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb_annotator.attach([render_product])

    # warm up the renderer so the first captured frames aren't blank
    for _ in range(10):
        sim.render()
        rgb_annotator.get_data()

    frames = []
    for step in range(motion.time_step_total):
        time_steps = torch.tensor([step], dtype=torch.long, device=sim.device)

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physics (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array(EYE_OFFSET), pos_lookat + np.array(LOOKAT_OFFSET))

        rgb_data = rgb_annotator.get_data()
        rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
        if rgb_data.size > 0:
            frames.append(rgb_data[:, :, :3].copy())

    import imageio

    imageio.mimwrite(args_cli.output, frames, fps=args_cli.fps, codec="libx264", quality=8)
    print(f"[INFO] Wrote {len(frames)} frames to {args_cli.output}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
