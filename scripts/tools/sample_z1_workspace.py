"""Pre-sample collision-free Z1 arm configurations on a fixed Aliengo base (phase 0).

Reproduces III-B of "Whole-Body End-Effector Pose Tracking" (doc/papers/Whole-Body_End-Effector_Pose_Tracking.md):
the base is pinned in the air with gravity off, the six arm joints are drawn uniformly inside their soft
limits, and a configuration is kept if no arm body reports a contact force (self-collision with itself,
the trunk or the legs) and the end-effector is not below the ground margin.

Each kept sample stores the arm joint angles (targets for the locomotion pre-training) and the
end-effector pose in the base frame (pose commands for the whole-body policy). The end-effector is the
``gripperStator`` frame; the gripper is held closed.

    python scripts/tools/sample_z1_workspace.py --headless
    -> datasets/aliengo_z1/z1_workspace.pt  (git-ignored)
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num_samples", type=int, default=10000, help="Collision-free samples to collect.")
parser.add_argument("--num_envs", type=int, default=4096, help="Configurations tested in parallel.")
parser.add_argument(
    "--min_ee_z", type=float, default=-0.32, help="Drop samples with the end-effector below this z [m, base frame]."
)
parser.add_argument("--contact_threshold", type=float, default=1e-3, help="Contact force counted as a collision [N].")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "aliengo_z1", "z1_workspace.pt"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from unitree_rl_lab.assets.robots.unitree import UNITREE_ALIENGO_Z1_CFG

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
ARM_BODIES = ["link01", "link02", "link03", "link04", "link05", "link06", "gripperStator", "gripperMover"]
EE_BODY = "gripperStator"
BASE_HEIGHT = 1.0  # [m] hang the robot in the air: no ground plane, nothing else to collide with


def _robot_cfg() -> ArticulationCfg:
    cfg = UNITREE_ALIENGO_Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.spawn = cfg.spawn.replace(fix_base=True, rigid_props=cfg.spawn.rigid_props.replace(disable_gravity=True))
    cfg.init_state = cfg.init_state.replace(pos=(0.0, 0.0, BASE_HEIGHT))
    return cfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _robot_cfg()
    # history_length=0: only the forces of the last physics step
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=0)


def main():
    torch.manual_seed(args_cli.seed)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    # 2 m apart: the arm reaches < 1 m, so neighbouring robots never touch
    scene = InteractiveScene(SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    sensor = scene["contact_forces"]
    arm_ids, arm_names = robot.find_joints(ARM_JOINTS, preserve_order=True)
    arm_sensor_ids, _ = sensor.find_bodies(ARM_BODIES, preserve_order=True)
    ee_id = robot.find_bodies(EE_BODY)[0][0]

    limits = robot.data.soft_joint_pos_limits[0, arm_ids]  # (6, 2)
    lo, hi = limits[:, 0], limits[:, 1]
    print(f"[INFO] arm joints {arm_names}")
    print(f"[INFO] soft limits lower {lo.tolist()}")
    print(f"[INFO] soft limits upper {hi.tolist()}")

    n = scene.num_envs
    default_q = robot.data.default_joint_pos.clone()  # legs standing, gripper closed
    zero_qd = torch.zeros_like(default_q)

    kept_q, kept_pos, kept_quat = [], [], []
    num_kept = num_tested = num_collision = num_low = 0
    while num_kept < args_cli.num_samples and simulation_app.is_running():
        q = default_q.clone()
        q[:, arm_ids] = lo + (hi - lo) * torch.rand(n, len(arm_ids), device=sim.device)
        robot.write_joint_state_to_sim(q, zero_qd)
        robot.set_joint_position_target(q)

        # Two physics steps: penetrating bodies report a contact force on the first step at the latest.
        body_hit = torch.zeros(n, len(arm_sensor_ids), dtype=torch.bool, device=sim.device)
        for _ in range(2):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            force = sensor.data.net_forces_w[:, arm_sensor_ids].norm(dim=-1)  # (n, bodies)
            body_hit |= force > args_cli.contact_threshold
        collided = body_hit.any(dim=-1)
        if num_tested == 0:
            rates = ", ".join(f"{b} {r:.0%}" for b, r in zip(ARM_BODIES, body_hit.float().mean(0).tolist()))
            print(f"[INFO] contact rate per body (first batch): {rates}")

        # Record the configuration that was tested (PD drift is negligible when nothing is touching).
        root_pose = robot.data.root_link_pose_w
        ee_pose = robot.data.body_link_pose_w[:, ee_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose[:, :3], root_pose[:, 3:7], ee_pose[:, :3], ee_pose[:, 3:7]
        )
        low = ee_pos_b[:, 2] < args_cli.min_ee_z
        ok = ~collided & ~low

        kept_q.append(q[ok][:, arm_ids].cpu())
        kept_pos.append(ee_pos_b[ok].cpu())
        kept_quat.append(ee_quat_b[ok].cpu())
        num_tested += n
        num_collision += int(collided.sum())
        num_low += int((low & ~collided).sum())
        num_kept += int(ok.sum())
        print(f"[INFO] kept {num_kept}/{args_cli.num_samples}  (tested {num_tested})")

    m = args_cli.num_samples
    data = {
        "arm_joint_names": arm_names,
        "arm_joint_pos": torch.cat(kept_q)[:m],  # (m, 6) [rad]
        "ee_pos_b": torch.cat(kept_pos)[:m],  # (m, 3) [m], base frame
        "ee_quat_b": torch.cat(kept_quat)[:m],  # (m, 4) (w, x, y, z), base frame
        "meta": {
            "ee_body": EE_BODY,
            "min_ee_z": args_cli.min_ee_z,
            "contact_threshold": args_cli.contact_threshold,
            "joint_pos_limits": limits.cpu(),
            "seed": args_cli.seed,
        },
    }
    output = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(data, output)

    pos = data["ee_pos_b"]
    print(
        f"[INFO] tested {num_tested}: collision {num_collision / num_tested:.1%},"
        f" below ground {num_low / num_tested:.1%}"
    )
    print(f"[INFO] ee_pos_b min {pos.min(0).values.tolist()}  max {pos.max(0).values.tolist()}")
    print(f"[INFO] saved {m} samples to {output}")


if __name__ == "__main__":
    main()
    # simulation_app.close() hangs here after the work is done; the dataset is already saved.
    os._exit(0)
