import gymnasium as gym

# Try-1 (2026-07-07): re-baseline check. Same reward weights as the old
# greedy-accumulation "BEST" (target_clearance=0.15, joint_pos=-0.2,
# flat_orientation_l2=-0.5, base_linear_velocity=-1.0, air_time_variance=-0.2,
# forward_command_progress=1.5, undesired_contacts without calf), but trained
# fresh from the Phase2 checkpoint (--load_run 2026-07-04_05-26-55
# --checkpoint model_6998.pt) under the corrected environment: demote_fraction
# reverted to the stock 0.5 (curriculum judgment itself is never modified --
# only rewards/weights are), and heading_command=True closes the
# backward-climbing exploit found during mujoco testing on 2026-07-06. Tests
# whether this reward design still reaches a good terrain_level without
# either crutch. See velocity_env_cfg_try1.py for the full reward listing.
#
# RESULT (2026-07-07, run 2026-07-07_01-23-09, 3000it): terrain_levels climbed
# steadily and plateaued at ~6.3-6.4 for the last ~2000 iterations (goal >6.0
# MET, cleanly this time -- no reversal exploit, no softened demote_fraction).
# entropy stayed healthy (~7.05, no collapse). Trade-off: Episode_Termination/
# bad_orientation rose from ~0.0015 in the old exploit-tainted regime to a
# stable 0.13-0.15 (peaked ~0.24 mid-run). time_out is still 78.5% of episodes
# so this is a real, survivable operating point, not a collapse -- but see
# Try-2 for an attempt to bring the fall rate down without giving back
# terrain_levels.
gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3-Try-1",
    entry_point="unitree_rl_lab.tasks.locomotion.robots.go2.velociy_en_go2:ManagerBasedRLEnvGo2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try1:RobotEnvCfgGo2Try1",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try1:RobotPlayEnvCfgGo2Try1",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_try1:SandboxPPORunnerCfg",
    },
)

# Try-2 (2026-07-07): same as Try-1, except heading_control_stiffness 1.0 -> 0.5
# (see velocity_env_cfg_try2.py docstring for the full evidence-grounded
# rationale: Try-1 logged error_vel_yaw~0.47rad persistently and an elevated
# base_angular_velocity penalty, both pointing at heading-correction torque
# fighting the stairs' natural pitch disturbance). Tests whether a gentler
# heading pull reduces bad_orientation (0.13-0.15 in Try-1) without giving back
# terrain_levels (~6.3 in Try-1).
#
# INTERRUPTED (2026-07-07, ~20min/3000it in, run 2026-07-07_06-58-25): killed
# to free the single GPU for Try-3, prioritized because Try-1 already meets the
# stated terrain_levels>6.0 goal, while Try-3 addresses a hard deployment
# blocker (see below). Revisit later if bad_orientation still needs work.
gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3-Try-2",
    entry_point="unitree_rl_lab.tasks.locomotion.robots.go2.velociy_en_go2:ManagerBasedRLEnvGo2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try2:RobotEnvCfgGo2Try2",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try2:RobotPlayEnvCfgGo2Try2",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_try2:SandboxPPORunnerCfg",
    },
)

# Try-3 (2026-07-07): same as Try-1, except the stair terrain's step_width
# (tread depth) is changed from a fixed 0.3 to a fixed 0.2, matching the real/
# mujoco target dimensions (see velocity_env_cfg_try3.py docstring for the full
# rationale). Direct mujoco testing of Try-1's export showed general stair
# climbing much smoother than before, but the height=20cm/width=20cm staircase
# specifically still gets the robot stuck shortly after it starts climbing --
# reproducing the original bug report. step_width has never been randomized or
# varied (always 0.3) unlike step_height, so 20cm tread depth was always
# out-of-distribution regardless of reward tuning. This trial deliberately
# tests a single narrower FIXED width first (not a randomized range yet --
# user wants to think through the allocation/proportions before doing that),
# and should also be checked for whether it still generalizes back UP to the
# no-longer-trained 0.3m tread.
#
# RESULT (2026-07-07, run 2026-07-07_07-17-39, 3000it): terrain_levels
# plateaued at ~5.4-5.5 (BELOW the >6.0 goal, and below Try-1's ~6.3) and
# Episode_Termination/bad_orientation settled around 0.27-0.30 (peaked ~0.36
# mid-run) -- both notably worse than Try-1 (6.3 / 0.13-0.15). entropy stayed
# healthy (~7.02, no collapse), so this is a real, harder-but-stable operating
# point, not a training failure. Narrowing the tread depth to 0.2m makes the
# AGGREGATE task (which also includes flat/rough/boxes terrain, unaffected by
# this change) measurably harder -- expected, since foot-placement timing on
# a shorter tread is objectively less forgiving. This aggregate regression
# does NOT by itself tell us whether the original mujoco failure (front foot
# catching on height=20/width=20 stairs) is fixed -- that specific skill may
# have improved a lot even while the aggregate numbers look worse. Exported
# and mujoco-deployed (2026-07-07); needs direct testing on both the 20cm-wide
# stairs (does it still get stuck?) and the original 30cm-wide ones (does it
# still climb them, now that it's never trained on that width?) before
# concluding anything from this run's tensorboard numbers alone.
gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3-Try-3",
    entry_point="unitree_rl_lab.tasks.locomotion.robots.go2.velociy_en_go2:ManagerBasedRLEnvGo2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try3:RobotEnvCfgGo2Try3",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try3:RobotPlayEnvCfgGo2Try3",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_try3:SandboxPPORunnerCfg",
    },
)

# Try-4 (2026-07-07): same as Try-1 (step_width back to 0.3, isolating this one
# variable from Try-3), except CommandsCfg.base_velocity.rel_heading_envs
# lowered 1.0 -> 0.3. See velocity_env_cfg_try4.py docstring for the full
# rationale: user reported, on BOTH Try-1 and Try-3 mujoco exports, that
# lateral (lin_vel_y) movement is broken even on flat ground, and the robot
# drifts diagonally instead of going straight while climbing. Hypothesis:
# heading_command at rel_heading_envs=1.0 means ang_vel_z is *never* a
# genuinely flat, sustained zero during training (always some small ongoing
# correction), unlike mujoco keyboard deploy which sends exactly that.
# Lowering rel_heading_envs restores real flat-zero-ang_vel_z training
# exposure for most envs, at the cost of weaker coverage against the
# reversal-climb exploit this whole heading_command change was meant to close.
#
# RESULT (2026-07-07, run 2026-07-07_19-30-56, 3000it): terrain_levels
# plateaued at ~6.1 (goal >6.0 MET, close to Try-1's ~6.3) and
# Episode_Termination/bad_orientation settled at ~0.12 -- actually slightly
# BETTER than Try-1's 0.13-0.15, not worse, despite only 30% of envs holding a
# target heading now. entropy stayed healthy (~7.00, comparable to Try-1's
# 7.05, no collapse). So on every tensorboard-visible metric this is at least
# as good as Try-1, with no apparent downside from lowering rel_heading_envs.
# error_vel_yaw is higher on aggregate (~0.52 vs Try-1's ~0.47) but this mixes
# heading-held and free envs and isn't directly comparable across the two
# settings. Exported and mujoco-deployed (2026-07-07); still needs direct
# testing to see whether lateral movement / straight-line drift are actually
# fixed -- tensorboard aggregates can't answer that, only hands-on testing
# can (same caveat as Try-3).
gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3-Try-4",
    entry_point="unitree_rl_lab.tasks.locomotion.robots.go2.velociy_en_go2:ManagerBasedRLEnvGo2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try4:RobotEnvCfgGo2Try4",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try4:RobotPlayEnvCfgGo2Try4",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_try4:SandboxPPORunnerCfg",
    },
)

# Try-5 (2026-07-07): STRATEGY CHANGE. heading_command is reverted globally
# (velocity_env_cfg.py's CommandsCfg.base_velocity is back to plain free
# ang_vel_z sampling, exactly as before this whole heading investigation).
# Instead, a new reward term mdp.heading_drift_penalty (weight=-0.3) tracks
# actual yaw against an "expected yaw" integrated from the COMMANDED ang_vel_z
# since spawn, and penalises the squared gap -- see rewards.py docstring for
# the full mechanism. Same as Try-1 otherwise (rewards, Phase2 origin,
# demote_fraction=0.5, step_width=0.3). Direct user testing showed
# heading_command (Try-1/3/4) never fixed lateral movement / straight-line
# drift and Try-4's rel_heading_envs=0.3 made general walking worse despite
# fine-looking tensorboard metrics -- user explicitly rejected continuing to
# stack command-generation patches and asked to try the reward-only route
# instead (this was Option B from the original 2026-07-06 discussion, not
# picked at the time in favor of heading_command).
#
# RESULT (2026-07-07, run 2026-07-07_23-17-06, 3000it): INVALIDATED BY A BUG.
# terrain_levels only reached ~4.5 (below every other Try, and below the >6.0
# goal), bad_orientation was ~47%, heading_drift_penalty averaged -0.69/step
# (huge). Root cause: the reward's reset-detection (`episode_length_buf==0`)
# never fires -- see velocity_env_cfg_try6.py / rewards.py for the fix and
# full explanation. The "expected yaw" tracker never reset across episode
# boundaries, so the robot was punished by comparing each fresh episode's
# random spawn yaw against a stale, unrelated previous-episode value --
# essentially random noise, not a real signal. This result says nothing about
# whether the underlying idea works; see Try-6 for the real test.
gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3-Try-5",
    entry_point="unitree_rl_lab.tasks.locomotion.robots.go2.velociy_en_go2:ManagerBasedRLEnvGo2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try5:RobotEnvCfgGo2Try5",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try5:RobotPlayEnvCfgGo2Try5",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_try5:SandboxPPORunnerCfg",
    },
)

# Try-6 (2026-07-07): re-test of Try-5 after fixing the episode_length_buf bug
# in mdp.heading_drift_penalty (see velocity_env_cfg_try6.py for the full
# explanation). Identical config to Try-5 otherwise.
#
# RESULT (2026-07-08, run 2026-07-08_09-58-37, 3000it): healthy this time --
# terrain_levels ~5.87 (just under the >6.0 goal, close to Try-1's 6.3),
# bad_orientation ~15.2% (back to Try-1-like levels, nowhere near Try-5's 47%
# bug-induced disaster), entropy ~7.01 (healthy), heading_drift_penalty
# averaging -0.16/step (much smaller and more plausible than Try-5's -0.69).
# Exported and mujoco-deployed (2026-07-08); still needs direct testing to
# confirm whether the reversal-climb behavior is actually gone and whether
# general walking quality (which heading_command had degraded) is intact.
gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3-Try-6",
    entry_point="unitree_rl_lab.tasks.locomotion.robots.go2.velociy_en_go2:ManagerBasedRLEnvGo2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try6:RobotEnvCfgGo2Try6",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try6:RobotPlayEnvCfgGo2Try6",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_try6:SandboxPPORunnerCfg",
    },
)

# Try-7 (2026-07-08): same as Try-6, plus a new mdp.feet_gait reward term
# (weight=0.2) enforcing a "lateral sequence walk" gait clock (one leg swings
# at a time, RR->FR->RL->FL, 75% stance duty factor). See
# velocity_env_cfg_try7.py docstring for the full rationale: user reported
# via direct mujoco testing of Try-6 that front feet place solidly on the
# next step but rear legs scramble to catch up ("バタバタ"), an uncoordinated
# front/rear timing issue that no existing reward term addresses (air-time
# terms only look at each leg in isolation, never cross-leg phase). This
# borrows the standard gait quadrupeds (dogs included) switch to specifically
# for stairs/uncertain footing, per the user's own suggestion to think about
# real dog stair-climbing gait. period=1.0s and weight=0.2 are first guesses,
# not evidence-derived -- watch Episode_Reward/feet_gait and actual mujoco
# behavior to tune in a Try-8 if needed.
#
# RESULT (2026-07-09, run 2026-07-08_20-52-09, 3000it): terrain_levels ~5.85
# (essentially unchanged from Try-6's 5.87 -- feet_gait doesn't cost climbing
# progress), bad_orientation ~10.2-10.4% (IMPROVED from Try-6's 15.2%, best
# since Try-4's 12%), entropy ~7.68 (healthy, no collapse), heading_drift_penalty
# ~-0.17/step (same as Try-6's -0.16, unaffected). Episode_Reward/feet_gait
# climbed to a stable ~0.34-0.36/step (out of a max plausible ~0.8 at this
# weight), confirming the policy actually learned some cross-leg phase
# coordination rather than the term going unused. Exported and mujoco-deployed
# (2026-07-09); still needs direct testing to confirm the rear-leg "バタバタ"
# scrambling is actually reduced and that lateral movement remains intact.
gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3-Try-7",
    entry_point="unitree_rl_lab.tasks.locomotion.robots.go2.velociy_en_go2:ManagerBasedRLEnvGo2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try7:RobotEnvCfgGo2Try7",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try7:RobotPlayEnvCfgGo2Try7",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_try7:SandboxPPORunnerCfg",
    },
)

# Try-8 (2026-07-09): hypothesis A. Adds mdp.body_height_gain (weight=1.0) on
# top of Try-7, rewarding the base directly for gaining absolute world height
# (dz/dt), independent of horizontal motion. See velocity_env_cfg_try8.py
# docstring for the full rationale: direct mujoco testing of Try-7 showed the
# robot gets physically stuck when front feet are staggered across two step
# heights on tall/narrow stairs -- it can't generate enough lift through the
# higher front leg to raise the body, so the rear feet never reach the step
# behind, and existing rewards (forward_command_progress = horizontal only,
# feet_gait = timing only) give zero signal during that straining moment.
# Tested in isolation against Try-9 (hypothesis B) -- both branch from Try-7
# independently.
#
# RESULT (2026-07-09, run 2026-07-09_05-46-45, 3000it): terrain_levels ~5.99
# (slightly above Try-7's 5.85, essentially unchanged), bad_orientation
# ~9.1-9.2% (IMPROVED from Try-7's 10.2-10.4%, best of any Try so far),
# entropy ~7.85 (healthy, no collapse). Episode_Reward/body_height_gain
# settled around 0.07-0.08/step -- present and stable, not zero, so the
# policy is using the signal, though the raw magnitude is modest relative to
# feet_gait's ~0.35/step. Exported and mujoco-deployed (2026-07-09); needs
# direct testing on the specific tall/narrow staggered-front-foot failure
# this was designed for.
#
# MUJOCO RESULT (2026-07-09): REJECTED. The robot became almost completely
# unable to move -- standing still turned into the effective optimum.
# Root cause: body_height_gain only rewards POSITIVE dz/dt (negative deltas
# are clamped to 0), so any small upward z jitter from contact/physics noise
# while standing still is pure free reward with no matching downside -- the
# exact "exp(-penalty) rewards not lifting at all" anti-pattern already
# documented and fixed once in foot_clearance_terrain_adaptive's docstring,
# recreated here in a new form. terrain_levels still looked fine in
# aggregate (4096-env average, where moving remains net-better once
# forward_command_progress etc. are included) but the exploit dominates in a
# single deployed instance, especially in exactly the hard tall/narrow-stair
# case this term was meant to fix, where genuinely climbing is risky/costly
# enough that farming jitter competes with attempting it. Hypothesis A is
# abandoned; a correct version would need to not reward any single positive
# step in isolation (e.g. require net height gain over a longer window, or
# use a strictly monotonic running-best design like Try-9's
# foot_touchdown_height_gain, which is structurally immune to this exploit).
gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3-Try-8",
    entry_point="unitree_rl_lab.tasks.locomotion.robots.go2.velociy_en_go2:ManagerBasedRLEnvGo2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try8:RobotEnvCfgGo2Try8",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try8:RobotPlayEnvCfgGo2Try8",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_try8:SandboxPPORunnerCfg",
    },
)

# Try-9 (2026-07-09): hypothesis B. Adds mdp.foot_touchdown_height_gain
# (weight=1.0) on top of Try-7 (NOT on top of Try-8 -- isolated from
# hypothesis A). Same motivating mujoco observation as Try-8, but rewards the
# discrete EVENT of any foot (most critically the rear foot that keeps
# failing to hook the step) touching down higher than its own best height so
# far this episode, instead of the base's continuous vertical velocity. See
# velocity_env_cfg_try9.py / rewards.py docstrings for the full mechanism
# (per-foot running-best tracking to prevent farming via bouncing).
#
# RESULT (2026-07-10, run 2026-07-09_18-51-56, 3000it): terrain_levels ~5.77
# (close to Try-7's 5.85, no regression), bad_orientation ~7.9-8.1% (best of
# any Try so far, better even than Try-8's 9.1-9.2%), entropy ~7.67 (healthy).
# BUT Episode_Reward/foot_touchdown_height_gain averaged only ~0.0001-0.0002/
# step -- three orders of magnitude smaller than feet_gait's ~0.35/step at a
# comparable weight=1.0. The reward only fires on a genuine new-max touchdown
# event (rare -- most steps are on flat/rough/box terrain or normal walking
# where no foot ever exceeds its episode-start height), and those rare spikes
# get diluted to near-zero once averaged into a per-step episodic reward
# stream. Practically, this term is not exploitable the way Try-8's was, but
# it also almost certainly had negligible influence on what the policy
# learned -- the close-to-Try-7 numbers (small bad_orientation improvement
# aside) are consistent with training having proceeded roughly as if this
# term were absent. So Try-9 does NOT meaningfully test hypothesis B as
# designed; a real test would need a much larger weight (to compensate for
# the sparsity) or a differently-scaled formulation. Exported and
# mujoco-deployed (2026-07-10) for hands-on testing regardless, but tensorboard
# numbers alone shouldn't be read as validating or refuting the underlying
# idea.
gym.register(
    id="Unitree-Go2-Velocity-v1-Phase3-Try-9",
    entry_point="unitree_rl_lab.tasks.locomotion.robots.go2.velociy_en_go2:ManagerBasedRLEnvGo2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try9:RobotEnvCfgGo2Try9",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_try9:RobotPlayEnvCfgGo2Try9",
        "rsl_rl_cfg_entry_point": f"{__name__}.velocity_env_cfg_try9:SandboxPPORunnerCfg",
    },
)
