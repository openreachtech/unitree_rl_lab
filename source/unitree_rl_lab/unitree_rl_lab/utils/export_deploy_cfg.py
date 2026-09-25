import numpy as np
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def _rename_observation_preserve_order(observations: dict, old_name: str, new_name: str) -> None:
    """Rename an observation key without moving it to the end of the dict (pop+assign would)."""
    if old_name not in observations:
        raise KeyError(
            f"Cannot rename observation '{old_name}' -> '{new_name}': "
            f"'{old_name}' not in exported observations {list(observations.keys())}"
        )
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

    # 2026-09-03: export the jump command's timing too. Without this the deploy side
    # (State_RLBase.cpp's jump_command / jump_time observations) silently falls back to
    # its own defaults -- jump_hold_time_s 1.0 s and max_jump_duration_s 1.5 s -- while
    # training has used max_jump_duration_s = 2.2 s since v5. The jump_time observation
    # is ``time_since_trigger / max_jump_duration_s``, so a 1.5 vs 2.2 mismatch feeds the
    # policy a phase signal 1.47x too large: at 0.3 s into a jump it is told 0.20 when
    # training would have said 0.14. Everything the policy times off that signal --
    # crucially when to reach the legs down -- is therefore early. tanaka's mujoco report
    # on v6b's model_5300 was "the height was tremendous, but it could not land, it fell
    # every time", which is what a mistimed landing looks like.
    if hasattr(env.cfg.commands, "jump_command"):
        jump_cfg = env.cfg.commands.jump_command
        cfg["commands"]["jump_command"] = {
            "max_jump_duration_s": float(jump_cfg.max_jump_duration_s),
            # NOT max_jump_duration_s: that is a timeout the training window almost
            # never reaches, since JumpCommand clears the command on landing. Holding
            # for the timeout leaves the deployed policy in jump mode long after it has
            # landed. See JumpCommandCfg.deploy_hold_time_s.
            "jump_hold_time_s": float(getattr(jump_cfg, "deploy_hold_time_s", 0.7)),
        }

        # 2026-09-06: export the orientation termination's relax window too. The deploy
        # FSM runs mdp::bad_orientation every control step and switches to Passive when
        # it fires, but training's bad_orientation_grounded suspends that check for
        # relax_window_s after the jump trigger -- a jump pitches past the limit on
        # essentially every attempt. Without this key the robot went limp mid-air and
        # toppled (tanaka's mujoco report on model_24600). State_RLBase.cpp defaults to
        # 1.0 s when the key is absent, so old exports keep working.
        orient_term = getattr(env.cfg.terminations, "bad_orientation", None)
        relax_window_s = getattr(orient_term, "params", {}).get("relax_window_s") if orient_term else None
        if relax_window_s is not None:
            cfg["commands"]["jump_command"]["orientation_relax_window_s"] = float(relax_window_s)

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
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
        if term_cfg.history_length == 0:
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

    # 2026-09-16: tuple を list に落としてから書く。
    # deploy.yaml を読むのは C++ の yaml-cpp と python の `yaml.safe_load` で、
    # どちらも `!!python/tuple` タグを読めない。tuple 型のフィールドを持つ項を1つ
    # 足しただけで（`DelayedJointPositionActionCfg.delay_steps_range`）、
    # mujoco 評価が全滅した（着地 0% が並び、壊れたポリシーと見分けがつかなかった）。
    def _no_tuples(v):
        if isinstance(v, tuple):
            return [_no_tuples(x) for x in v]
        if isinstance(v, list):
            return [_no_tuples(x) for x in v]
        if isinstance(v, dict):
            return {k: _no_tuples(x) for k, x in v.items()}
        return v

    cfg = _no_tuples(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
