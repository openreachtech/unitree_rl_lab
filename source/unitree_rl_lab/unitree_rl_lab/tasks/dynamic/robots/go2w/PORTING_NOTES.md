# Go2w jump/flip tasks — porting notes

These tasks (`Go2w-Jump-Phase1`, `Go2w-Jump-Phase2`) are a port of the go2 EFGCL
jump/backflip/sideflip tasks from `origin/feat/jump`
(`tasks/dynamic/robots/go2/`) to the **wheeled** Go2w. The approach follows the paper
`doc/papers/EFGCL_Learning_Dynamic_Motion_through_Spotting-Inspired_External_Force_Guided_Curriculum_Learning.md`:
a one-shot jump command triggers a brief external **assist force** (the "spotter"),
and a curriculum decays that force to zero as the policy succeeds on its own.

- **Phase 1** (`jump_env_cfg_phase1.py`) — quiet standing with the final jump
  observation layout. A warm-start for Phase 2 (`--previous-task Go2w-Jump-Phase1
  --resume`).
- **Jump** (`jump_env_cfg_jump.py`, `Go2w-Jump`) — vertical jump only, with go2's
  `Go2-Jump-60` reward fixes, a measured standing height (0.405), a horizontal-drift
  penalty for the wheels, overshoot-tolerant success and a 0.005 assist decay. The
  recipe base for the two flips.
- **Backflip** (`jump_env_cfg_backflip.py`, `Go2w-Backflip`) — Jump's recipe, one turn of
  pitch: 0.30 m launch + 200 N post-take-off pitch couple on top of the front-hip force.
- **Sideflip** (`jump_env_cfg_sideflip.py`, `Go2w-Sideflip`) — Jump's recipe, one turn of
  roll: 0.50 m launch + 65 N*m whole-body spin. go2's hip-force assist cannot roll go2w
  (roll inertia 4x go2's, sitting in the heavy calves and wheels); see that file.
- **Phase 2** (`jump_env_cfg_phase2.py`, `Go2w-Jump-Phase2`) — the three above in one
  policy, one motion sampled per environment per episode, every per-motion setting read
  from the single-motion configs.

All of them resume from Phase 1 (`--previous-task Go2w-Jump-Phase1 --resume`).

The robot-agnostic machinery (`../../mdp/`, `../../agents/`) started as a copy of the
go2 version — `JumpCommand` and its force/curriculum logic operate on body names
(`FR_hip`, …), root state, and projected gravity, none of which depend on joint count.
It has since diverged: go2w added per-flip heights, the backflip pitch couple, the
whole-body sideflip spin, overshoot-tolerant success and the assist-force arrows, and
dropped go2-only experiment code this branch never used (sideflip hip couple,
`flip_launch_height`, and the `jump_progress`/`landing_impact`/`flip_forward_axis_tilt`/
windup-standing rewards).

## What was adapted for the wheeled robot

Go2w has **16 joints**: the 12 leg joints (`*_hip/thigh/calf_joint`) plus 4 wheel
joints (`*_foot_joint`). The leg joints are position-controlled; the wheels are
velocity-controlled continuous joints. This mirrors the established split in
`locomotion/robots/go2w/velocity_env_cfg.py` (`LEG_JOINT_NAMES` / `WHEEL_JOINT_NAMES`).

1. **Robot asset** — `UNITREE_GO2W_CFG` instead of `UNITREE_GO2_CFG`.
2. **Actions** — split into `JointPositionAction` (legs, scale 0.25) and
   `JointVelocityAction` (wheels, scale 8.0), instead of one position action over
   `.*`. The wheels stay in the action space (the policy may brake or spin them during
   a flip), matching the velocity task and the deploy config. The term names are
   load-bearing for `export_deploy_cfg`.
3. **Observations** — `joint_pos_rel` is scoped to the legs (a wheel's absolute angle
   drifts without bound and carries no usable pose information). `joint_vel_rel` stays
   over all joints (bounded and informative).
4. **Pose/limit rewards** — `standing_pose`, `pre_jump_pose`,
   `motion_progress_standing`, `joint_vel_l2`, `joint_acc_l2`, `joint_torques_l2`,
   `joint_pos_limits` are all scoped to `LEG_JOINT_NAMES`. The two pose-sum rewards in
   `../../mdp/rewards.py` (`standing_pose_reward`, `motion_progress_standing_reward`)
   were changed to honour `asset_cfg.joint_ids` so this scoping takes effect; the
   default (`slice(None)` → all joints) keeps the legged go2 behaviour identical.
5. **Standing height** — `nominal_standing_height = 0.405`, measured with the Phase 1
   policy standing (the base config's 0.45 is the spawn height).
6. **Contacts** — undesired-contact bodies remain `Head_.*`, `.*_hip`, `.*_thigh`,
   `.*_calf`. The `.*_foot` wheels are the legitimate ground-contact bodies and are
   never penalised (same as go2's toes).
7. **Registration** — `scripts/list_envs.py` now also walks `dynamic.robots` so
   `train.py` offers these task IDs.

Train Phase 1 first, then resume each motion task (or Phase 2) from it.
