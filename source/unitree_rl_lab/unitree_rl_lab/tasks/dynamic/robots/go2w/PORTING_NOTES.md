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
- **Phase 2** (`jump_env_cfg_phase2.py`) — unified assisted **jump + backflip +
  sideflip**, one motion sampled per environment per episode. `TRAIN_JUMP /
  TRAIN_BACKFLIP / TRAIN_SIDEFLIP` at the top of the file toggle which motions are in
  the mix.

The robot-agnostic machinery (`../../mdp/`, `../../agents/`) is a verbatim copy of the
go2 version — `JumpCommand` and all its force/curriculum logic operate on body names
(`FR_hip`, …), root state, and projected gravity, none of which depend on joint count.

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
5. **Standing height** — `JumpCommandCfg.nominal_standing_height = 0.45` (go2w spawns
   at z=0.45 vs go2's 0.40).
6. **Contacts** — undesired-contact bodies remain `Head_.*`, `.*_hip`, `.*_thigh`,
   `.*_calf`. The `.*_foot` wheels are the legitimate ground-contact bodies and are
   never penalised (same as go2's toes).
7. **Registration** — `scripts/list_envs.py` now also walks `dynamic.robots` so
   `train.py` offers these task IDs.

## Likely retuning points (assist/curriculum were tuned on go2)

The force magnitudes, target height, and curriculum thresholds in `CommandsCfgPhase2`
were tuned for the lighter, wheel-less go2 and are a starting point, not a solution:

- `target_height_range=(0.20, 0.20)` and the `backflip/sideflip/crouch_assist_force`
  values assume go2's mass; go2w carries extra wheel/hub mass, so the launch forces
  likely need to go up to clear the ground and rotate.
- Landing on wheels behaves differently from landing on toes (they roll). Watch
  `minimum_landing_time_s`, the landing tolerances, and whether the policy exploits
  wheel roll to "cheat" the upright/pose check.
- `jump_assist_mass` auto-detects the simulated total mass, so the projectile-derived
  jump force self-adjusts; the flip forces are fixed constants and do not.

Train Phase 1 first, then resume Phase 2 from it, exactly as on go2.
