# Go2w biped task -- porting notes

`Go2w-Biped-Front` (`biped_env_cfg_front.py`) is a port of go2's `Go2-Multitask-Biped-Front`
from `origin/feat/multi-task` (`tasks/biped/robots/go2/biped_front_env_cfg_multitask.py`): the
robot rises onto its front legs, hind legs tucked, and follows a velocity command there.
`Go2w-Biped-Hind` (`biped_env_cfg_hind.py`) is its mirror, ported from `Go2-Multitask-Biped-Hind`:
`stance = STANCE_HIND`, every side-naming term restated for the other pair, and
`front_hip_height` dropped (the front hips are the raised end there). Both train from scratch.
`Go2w-Biped` (`biped_env_cfg.py`) is both in one policy, ported from `Go2-Multitask-Biped`: the
stance is drawn per episode and observed through `handstand_command`; see that file for the
training procedure (start from the hind policy, balance the stance population, front-only
`front_hip_height`). The deployed checkpoint was trained in two stages, before step 3 existed:
`logs/rsl_rl/go2w_biped/2026-10-15_22-04-10` (steps 1-2, 1500 it from `0000_init_from_hind`),
resumed as `2026-10-16_12-49-14` (+ step 3, 1000 it). The folded config reproduces the second.

It is ported as a **standalone skill**. go2's version is an expert for a mixture-of-experts policy
and sits on `tasks/multitask` (a 122-column observation shared with the jump and gait experts, an
idle jump command, `gated` reward wrappers keyed on command state). None of that is carried over.

## What came from `feat/multi-task`

- `../../mdp/stance_rewards.py` -- unchanged (CoM-CoP balance terms, stance pitch/roll, upright
  balance, support polygon, front-hip height, lifted-foot contact, success).
- `../../mdp/handstand.py` -- `HandstandCommand` cut down to a stance held for the whole episode:
  no scheduling, release, descent or take-off gate. Pinned for the single-stance tasks (not
  observed there); drawn per episode for `Go2w-Biped`, with the per-stance success metrics and a
  new `balance_stance_population` option. Supplies the posture quantities and `handstand/*`.
- `../../mdp/gating.py` -- the "is upright" gate (`when_upright`, scaling by
  `HandstandCommand.upright_ramp`), `stance_aware` (go2's per-stance router), the new
  `when_stance`, and the startup resolver for `SceneEntityCfg`s nested in wrapper params. Terms
  go2 gated on "stance commanded" are plain terms here: the stance is always commanded.
- Reward weights, contact pricing, the -200 termination penalty, 0-30 ms actuation delay and the
  spawn-height sampling are go2's.

## What was adapted for the wheeled robot

1. **Rolling, not stepping** -- go2's `feet_air_time` (two-foot stepping cadence) is dropped. In a
   nose-down stance Go2w's front wheel axles point sideways, so it balances and travels fore-aft by
   driving them.
2. **Velocity command** -- fore-aft +-1.0 m/s and yaw +-1.0 rad/s, `lin_vel_y` = 0 (a wheel cannot
   roll sideways; go2's command was +-1.0 m/s in both axes, +-0.5 lateral for the hind stance).
   This also removes the hind stance's two open go2 problems -- the tripod shuffle and
   sidestepping -- which were stepping-gait problems.
3. **Joint penalties** -- `joint_vel`/`acc`/`torques`, `energy`, `dof_pos_limits` on the 12 leg
   joints only. New `lifted_wheel_vel` (-1e-3) stops the lifted pair's wheels spinning in the air.
4. **Heights** -- wheel radius 0.084 m (wheel-centre resting height), reach 0.213 + 0.2264 + 0.084
   = 0.524 m against go2's 0.448. Front-hip target 0.30 -> 0.35, base target 0.55 -> 0.65, spawn
   drop 0.078 -> 0.045 (0.45 spawn - 0.405 standing).
5. **Robot, actions, events** -- the Go2w jump tasks' scene and actuator model (`ACTUATORS` in
   `dynamic/robots/go2w/jump_env_cfg.py`, with the delay added), leg position + wheel velocity
   actions, and the Go2w velocity task's randomisation.
6. **Observations** -- the Go2w layout (leg-only `joint_pos_rel`, all-joint `joint_vel_rel`) with
   history 3 and the velocity command; the critic adds base velocity, root height, CoM-CoP and
   joint effort.

## Known risk

Go2w weighs 19.7 kg against go2's ~15, the extra mass in the wheels at the leg tips. go2's rise
already held the stance shoulders at the 20.2 N*m ceiling. If the rise stalls, the jump tasks'
assist-force curriculum (lifting the rear) is the fallback.

It did not stall: both single-stance tasks rise without assistance.

## Deploy

`deploy/robots/go2w`: `State_Biped`, keyboard workflow as Go2's on `feat/biped`. From FixStand,
[3] hind wheels, [4] front wheels, both running `Go2w-Biped` with a fixed `handstand_command`
stance; [f]/[b]/[y]/[u] velocity, [0] Passive. Checked in MuJoCo.
