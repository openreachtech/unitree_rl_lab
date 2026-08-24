# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--pin_vel_x",
    type=float,
    default=None,
    help=(
        "Pin the forward velocity command to this value (m/s) for every env, with lateral and yaw "
        "commands zeroed and no standing envs. Without it, the play cfg samples commands over the "
        "task's whole limit range, so watching a fast policy means waiting for a fast draw."
    ),
)
parser.add_argument(
    "--telemetry-csv",
    type=str,
    default=None,
    help=(
        "If set, log per-step telemetry (jump command state, joint velocity/torque) for env 0 to this "
        "CSV path -- same columns as the deploy-side State_Flip telemetry, for direct sim2sim comparison. "
        "Only meaningful for jump/dynamic tasks with a 'jump' command term."
    ),
)
parser.add_argument(
    "--telemetry-steps",
    type=int,
    default=250,
    help="Number of steps to capture before exiting automatically when --telemetry-csv is set.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    if args_cli.pin_vel_x is not None:
        v = args_cli.pin_vel_x
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (v, v)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        print(f"[INFO] forward velocity command pinned to {v} m/s on every env")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # --- optional diagnostic telemetry (sim2sim gap investigation) ---
    # Mirrors the deploy-side State_Flip telemetry columns (see State_Flip.cpp) for env 0,
    # so a MuJoCo run can be directly compared against an Isaac Sim play-mode trace.
    telemetry_file = None
    telemetry_joint_names = [
        "FL_hip", "FR_hip", "RL_hip", "RR_hip",
        "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
        "FL_calf", "FR_calf", "RL_calf", "RR_calf",
    ]
    if args_cli.telemetry_csv:
        telemetry_file = open(args_cli.telemetry_csv, "w")
        header = ["t", "enabled", "elapsed_since_trigger", "accumulated_pitch_deg", "grav_x", "grav_y", "grav_z"]
        header += [f"{n}_vel" for n in telemetry_joint_names]
        header += [f"{n}_tau" for n in telemetry_joint_names]
        telemetry_file.write(",".join(header) + "\n")

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    telemetry_step = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if telemetry_file is not None:
            base_env = env.unwrapped
            asset = base_env.scene["robot"]
            command = base_env.command_manager.get_term("jump")
            row = [
                telemetry_step * dt,
                float(command.enabled[0]),
                float(command.elapsed_since_trigger[0]),
                float(command.accumulated_pitch[0]) * 180.0 / 3.141592653589793,
                float(asset.data.projected_gravity_b[0, 0]),
                float(asset.data.projected_gravity_b[0, 1]),
                float(asset.data.projected_gravity_b[0, 2]),
            ]
            row += [float(v) for v in asset.data.joint_vel[0].tolist()]
            row += [float(v) for v in asset.data.applied_torque[0].tolist()]
            telemetry_file.write(",".join(str(x) for x in row) + "\n")
            telemetry_file.flush()
            telemetry_step += 1
            if telemetry_step >= args_cli.telemetry_steps:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if telemetry_file is not None:
        telemetry_file.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
