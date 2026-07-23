import numpy as np
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def _rename_observation_preserve_order(observations: dict, old_name: str, new_name: str) -> None:
    """Rename an observation key without moving it to the end of the dict (pop+assign would).

    A no-op if ``old_name`` is absent: renames like ``velocity_commands`` are
    velocity-task-specific and legitimately don't apply to other task families
    (e.g. jump/backflip use ``jump_command`` instead).
    """
    if old_name not in observations:
        return
    items = list(observations.items())
    observations.clear()
    observations.update({(new_name if key == old_name else key): value for key, value in items})


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def export_deploy_cfg(
    env: ManagerBasedRLEnv,
    log_dir,
    observation_renames: dict[str, str] | None = None,
):
    asset: Articulation = env.scene["robot"]
    joint_sdk_names = env.cfg.scene.robot.joint_sdk_names
    joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)

    cfg = {}  # noqa: SIM904
    cfg["joint_ids_map"] = joint_ids_map
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    stiffness = np.zeros(len(joint_sdk_names))
    stiffness[joint_ids_map] = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    cfg["stiffness"] = stiffness.tolist()
    damping = np.zeros(len(joint_sdk_names))
    damping[joint_ids_map] = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    cfg["damping"] = damping.tolist()
    cfg["default_joint_pos"] = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    # --- actions ---
    # Keyed by the action term's actual class name (e.g. "JointPositionAction"), not its
    # ActionsCfg attribute name -- the deploy-side C++ registry (REGISTER_ACTION) looks
    # actions up by class name, so an attribute named e.g. "joint_position" (as in the
    # jump/dynamic tasks) would export an unrecognized key and crash go2_ctrl at startup.
    # This previously only worked for tasks (like velocity) whose ActionsCfg attribute
    # happened to be named identically to the class itself.
    action_terms = env.action_manager._terms.values()
    cfg["actions"] = {}
    for action_term in action_terms:
        action_name = type(action_term).__name__
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            del term_cfg[_]
        cfg["actions"][action_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_name]["joint_ids"] = None
        else:
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

    # --- observations ---
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    for obs_name, obs_cfg in obs_terms:
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            scale = term_cfg.scale.detach().cpu().numpy().tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length > 0:
            term_cfg.history_length = int(term_cfg.history_length)
        else:
            term_cfg.history_length = 1

        # clean cfg
        term_cfg = term_cfg.to_dict()
        for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
            del term_cfg[_]
        cfg["observations"][obs_name] = term_cfg

    if observation_renames:
        for old_name, new_name in observation_renames.items():
            _rename_observation_preserve_order(cfg["observations"], old_name, new_name)

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
