"""Two-stage get-up training in one run (see ``mdp.ramp_posture_rewards``).

``make_staged(cfg, host_params)`` turns an env cfg with the posture terms into one that starts
with HoST's reward set: the posture terms start at weight 0 and the listed params at HoST's
values; once the get-up curriculum is done they are ramped to the cfg's own (final) values.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm

from . import mdp as getup_mdp

# terms added on top of HoST to shape the standing posture
POSTURE_TERMS = (
    "style_kneeling", "style_feet_width", "style_hip_yaw_l1", "style_hip_roll_l1",
    "target_lower_body_pose", "target_leg_pose_l2", "target_arm_pose_l2",
    "target_stand_base_vel", "target_stand_feet_vel", "target_stand_joint_vel",
)
STEPS_PER_ITERATION = 50


def make_staged(cfg, host_params: dict[str, object], ramp_iterations: int = 1000, delay_iterations: int = 0) -> None:
    r = cfg.rewards
    final_weights = {}
    for name in POSTURE_TERMS:
        term = getattr(r, name, None)
        if term is not None:
            final_weights[name] = term.weight
            term.weight = 0.0
    final_params = {}
    for key, value in host_params.items():
        name, param = key.split("/")
        term = getattr(r, name)
        final_params[key] = term.params[param]
        term.params[param] = value
    cfg.curriculum.posture = CurrTerm(
        func=getup_mdp.ramp_posture_rewards,
        params={
            "final_weights": final_weights,
            "host_params": dict(host_params),
            "final_params": final_params,
            "ramp_steps": ramp_iterations * STEPS_PER_ITERATION,
            "delay_steps": delay_iterations * STEPS_PER_ITERATION,
        },
    )
