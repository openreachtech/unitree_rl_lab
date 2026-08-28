"""Roll out a GO2 DMP motion primitive (from TillHielscher/gen-mod-expressive-motion-behavior /
animation-dmp) through Isaac Sim's forward kinematics and save it as a MotionLoader-compatible npz,
in the same format produced by scripts/mimic/csv_to_npz.py for G1.

.. code-block:: bash

    python go2_dmp_to_npz.py -f path/to/robot_go2_primitives/shake_hand --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Roll out a GO2 DMP primitive and save it as a mimic-ready npz.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="Path to the primitive (no .json/_weights.npy suffix).")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument("--output_name", type=str, help="The name of the output npz file.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.output_name:
    args_cli.output_name = args_cli.input_file + ".npz"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from animation_dmp import DMP

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(scene.cfg.robot.joint_sdk_names, preserve_order=True)[0]

    # roll out the DMP directly at the target output fps
    dmp = DMP.load(args_cli.input_file)
    dmp.dt = 1.0 / args_cli.output_fps
    dof_pos = dmp.run()  # (n_frames, 12), already in joint_sdk_names order
    dof_vel = np.gradient(dof_pos, dmp.dt, axis=0)
    n_frames = dof_pos.shape[0]

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    root_states = robot.data.default_root_state.clone()
    root_states[:, :2] += scene.env_origins[:, :2]

    for i in range(n_frames):
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = torch_from(dof_pos[i], joint_pos.device)
        joint_vel[:, robot_joint_indexes] = torch_from(dof_vel[i], joint_vel.device)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()  # kinematic replay only, no physics step
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)

    np.savez(args_cli.output_name, **log)
    print(f"[INFO]: Rolled out {n_frames} frames ({n_frames / args_cli.output_fps:.2f} sec)")
    print("[INFO]: Motion npz file saved to", args_cli.output_name)


def torch_from(arr, device):
    import torch

    return torch.tensor(arr, dtype=torch.float32, device=device)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
