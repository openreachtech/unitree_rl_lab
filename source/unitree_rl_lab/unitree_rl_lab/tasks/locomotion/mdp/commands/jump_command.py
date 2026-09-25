from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class JumpCommand(CommandTerm):
    """Binary jump-trigger command for the running long-jump task (Stage B).

    See リファレンス_ExplicitEstimator実装仕様.md and
    リファレンス_キーボード操作とNet2Net設計の下調べ.md for the original design, and
    プロジェクト_StageB崩壊の原因分析.md for the 2026-09-02 rewrite this file now
    reflects.

    - Starts at 0 ("running mode").
    - At each resample boundary (``cfg.resampling_time_range``), envs currently at 0
      get a coin flip (``cfg.jump_probability``) to trigger a jump (-> 1). Envs
      already at 1 (mid-jump) are left untouched.
    - Once triggered it stays at 1 until EITHER a genuine landing is detected OR
      ``cfg.max_jump_duration_s`` elapses (hard timeout).
    - On real-robot deployment this bit is driven by the keyboard's one-shot
      ``consume_jump_trigger()`` instead of the random coin flip.

    2026-09-02 REWRITE (three fixes, all diagnosed from the Stage B v2 collapse):

    1. **Landing detection required an airborne phase.** The old condition was
       ``all_feet_down & upright & (time_since_trigger > min_air_time_s)`` with
       ``min_air_time_s = 0.05`` (2.5 control steps). A robot that was simply
       standing satisfied all three ~0.06 s after the trigger, so the command
       cleared itself before any jump could be attempted -- the policy never
       received an actionable jump signal at all (observed
       ``jumping_fraction`` 1-4% for the whole of two 10k/3k-iteration runs, and
       ``jump_vel_target_levels`` never moving off its initial value). Landing now
       additionally requires ``has_been_airborne``: all four feet must actually
       have left the ground at some point since the trigger.

    2. **A fallen robot stayed "jumping" forever.** ``upright`` was part of the
       landing condition, so a robot lying on its side never satisfied it and kept
       ``jump_command == 1`` indefinitely. Combined with the jump-gated
       ``bad_orientation_grounded`` termination, falling over bought permanent
       immunity from the orientation termination. ``max_jump_duration_s`` now
       force-clears the command regardless of pose, so that exploit is closed
       from this side as well as from the termination side.

    3. **The jump was never physically demonstrated.** Pure reward shaping has to
       discover a ballistic push-off by chance. Following tak's ``Go2-Jump-60``
       (tag ``jump-demo``, which does land real jumps) this now applies the EFGCL
       external-force assist: a brief downward crouch pulse on all four hips at
       the trigger, immediately followed by a ramped vertical launch force sized
       from projectile motion for ``cfg.assist_apex_height_m``. ``assist_scale``
       decays toward 0 as the policy succeeds on its own
       (mdp.jump_assist_decay), so the assist is a teacher, not a crutch.
       Reference: [[reference_paper_efgcl_external_force_guided_curriculum]].
    """

    cfg: JumpCommandCfg

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.cfg.contact_sensor_cfg.resolve(env.scene)

        self.robot: Articulation = env.scene[cfg.asset_cfg.name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.contact_sensor_cfg.name]

        self.jump_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.time_since_trigger = torch.zeros(self.num_envs, device=self.device)
        # Whether all four feet have left the ground at any point since the trigger.
        # Landing detection requires this -- see fix (1) in the class docstring.
        self.has_been_airborne = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_all_airborne = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Per-episode outcome for the assist-decay curriculum: a jump that both left
        # the ground and came back down upright.
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Per-episode "a real jump happened", landing quality not required. This is the
        # signal mdp.jump_assist_decay gates on from v5 onward -- see _update_command.
        self.episode_real_jump = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- jump measurement (2026-09-03) ------------------------------------------
        # Added because the previous run had no way to tell a jump from a fast run.
        # ``has_been_airborne`` alone is satisfied by the suspension phase every
        # quadruped gait has above ~3 m/s, so it reported 100% "airborne" while mujoco
        # showed the robot barely leaving the ground. Height and airborne duration are
        # what actually separate the two, and the reward set needs the numbers anyway.
        #
        # Reference height is captured per jump at the trigger instead of a hardcoded
        # nominal standing height: it self-calibrates, and it avoids the class of bug
        # tak documented where one wrong standing-height constant (0.40 vs 0.30) fed
        # three separate quantities at once.
        self.trigger_height = torch.zeros(self.num_envs, device=self.device)
        self.max_height_gain = torch.zeros(self.num_envs, device=self.device)
        self.airborne_time = torch.zeros(self.num_envs, device=self.device)
        self._liftoff_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)
        # 2026-09-03 (v5): the liftoff position must be latched at the FIRST time all four
        # feet leave the ground in a jump window, not re-latched at every flight phase.
        # The v4 code overwrote it on each ``newly_airborne`` edge, so a jump followed by
        # a couple of running strides inside the same window measured only the last hop:
        # logged jump_distance read 0.06-0.15 m while the real displacement was metres.
        self._liftoff_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 2026-09-03 (v6), both added from tanaka's mujoco observation of model_3300:
        # "it hops, then bounces like a ball -- about three vertical oscillations per
        # jump". The metrics said the same thing: airborne_time 0.72 s against the
        # 0.277 s a single ballistic 0.094 m jump would take, i.e. ~2.6 flights per
        # window. See プロジェクト_StageB着地失敗の原因分析.md.
        #
        # ``liftoff_count`` counts flight phases so mdp.jump_repeat_liftoff can charge
        # for the 2nd and later ones. ``touched_down_once`` marks the end of the first
        # descent, so mdp.jump_trunk_clearance can stop demanding a high trunk exactly
        # when the legs should be absorbing the landing.
        self.liftoff_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 2026-09-03: velocity of the base at the instant of first liftoff. Added to
        # settle where the jump distance is being lost. distance = 2*v_x*v_z/g, and
        # back-solving the two policies we have measured gives v_x = 1.72 m/s for v5's
        # model_4300 (h 0.069 m, d 0.408 m) against 1.16 m/s for v6b's model_5300
        # (h 0.081 m, d 0.298 m) -- i.e. the taller jump appears to be bought by braking
        # harder, keeping only 35% of the 3.3 m/s approach against v5's 52%. These two
        # metrics measure that directly instead of inferring it.
        self.liftoff_vel_x = torch.zeros(self.num_envs, device=self.device)
        self.liftoff_vel_z = torch.zeros(self.num_envs, device=self.device)
        # 2026-09-04 (Loop 10 diagnosis): WHICH flight phase of the window turned out to
        # be the take-off, 1-based. Added as instrumentation to settle a question the
        # reward budget raised: mdp.jump_repeat_liftoff charged (liftoff_count - 1) per
        # step, and at a 3.3 m/s approach an ordinary running stride produces a flight
        # phase, so if the take-off is habitually the 2nd or 3rd liftoff of the window
        # then that penalty is being levied on the jump itself rather than on a bounce.
        #
        # It was: measured on v8's model_7600 the take-off was the 2nd liftoff in every
        # one of five sampled iterations (unaided_liftoff_index = 2.0000), so the penalty
        # was charged for the whole of every real jump. 2026-09-05 (Loop 10): this is no
        # longer instrumentation only -- mdp.jump_repeat_liftoff now counts surplus
        # flights from this index instead of from the start of the window.
        self.liftoff_index = torch.zeros(self.num_envs, device=self.device)
        # 2026-09-05 (Loop 10 instrumentation): where the approach's kinetic energy goes.
        #
        # The Loop 9 post-mortem showed the binding constraint is the take-off energy
        # C = v_x^2 + v_z^2 (4.13 m^2/s^2 measured) against the approach's 2*KE/m of
        # 10.89 at 3.3 m/s -- 62% of it is lost. Raising C is worth up to 3.1x on
        # distance, against +4% for any further rebalancing of the v_x/v_z split, so it
        # is the only lever left worth pulling. But "lost" does not say WHERE, and the
        # two candidates want opposite fixes:
        #
        #   (i) the robot decelerates over several strides before it jumps (a gather).
        #       Fixable by shaping: the approach itself is being run too slowly.
        #   (ii) the energy goes into the take-off stance -- the plant that redirects
        #       horizontal into vertical. Not fixable by shaping; it is leg mechanics and
        #       take-off technique, and would call for the EFGCL route instead.
        #
        # Three speeds separate them: at the trigger (still running), at the touchdown
        # that begins the take-off stance, and at lift-off. Instrumentation only -- no
        # reward reads these, so Loop 10's reward change stays a single line.
        self.trigger_speed = torch.zeros(self.num_envs, device=self.device)
        self.takeoff_stance_entry_speed = torch.zeros(self.num_envs, device=self.device)
        self._last_touchdown_speed = torch.zeros(self.num_envs, device=self.device)
        self.touched_down_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Seconds since that first touchdown, so mdp.jump_landing_feet_first can be
        # confined to the landing itself instead of paying for the whole rest of the
        # window -- see that function for what went wrong without this.
        self.time_since_touchdown = torch.zeros(self.num_envs, device=self.device)
        # One-step pulse at a scoring landing, read by mdp.jump_distance_reward.
        self.jump_distance = torch.zeros(self.num_envs, device=self.device)
        # Persists until the next jump, for the logged metric.
        self.last_jump_distance = torch.zeros(self.num_envs, device=self.device)
        # A jump that cleared both thresholds -- i.e. not a running suspension phase.
        self.real_jump = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- EFGCL assist state ---
        self.assist_scale: float = float(cfg.initial_assist_scale)
        self.curriculum_success_rate: float = 0.0
        self.curriculum_episode_count: int = 0
        self.curriculum_success_count: int = 0
        self.curriculum_real_jump_count: int = 0

        # 2026-09-03 (v5): a fixed subset of envs NEVER receives the assist, so every
        # logged number has an honest, assist-free counterpart.
        #
        # This closes the measurement hole that made v4 look like a success. The assist's
        # launch force is sized (see _apply_assistance) to impart exactly
        # v0 = sqrt(2*g*assist_apex_height_m) by itself, and v4 set
        # assist_apex_height_m == the reward's target_height == 0.20 m. So the teacher
        # alone produced the full target jump, the reward was satisfied without the policy
        # contributing anything, and every metric (max_height_gain 0.21 m,
        # real_jump_fraction 90%) was measuring the external force rather than the
        # policy. In mujoco, where no assist exists, tanaka observed almost no jump at
        # all. Same class of bug as v3's airborne_fraction: a metric that measures
        # something other than what it is named after.
        holdout = torch.rand(self.num_envs, device=self.device) < cfg.assist_holdout_fraction
        # Guarantee the holdout group is non-empty even for tiny num_envs (smoke tests).
        if self.num_envs > 0 and not bool(holdout.any()):
            holdout[0] = True
        self.assist_holdout = holdout

        self._assist_body_ids, _ = self.robot.find_bodies(list(cfg.assist_body_names))
        # Total robot mass, for sizing the launch force from projectile motion.
        self._robot_mass = float(self.robot.data.default_mass[0].sum().item())

        # --- 2026-09-06 (Loop 13): the four numbers tanaka's mujoco description needs.
        # "The front legs still think they are flying, the rear legs touch down at once,
        # and the front legs get slammed into the ground" is not visible in any metric
        # logged so far: max_height_gain, airborne_time and success_rate are all blind to
        # the ATTITUDE of the jump and to the ORDER the feet arrive in.
        #
        # peak_foot_clearance is also the honest answer to "it does not look very high":
        # max_height_gain is the TRUNK's rise from the trigger height, which is not what a
        # person watching sees. What they see is daylight under the feet.
        self._foot_link_ids, _foot_link_names = self.robot.find_bodies([".*_foot"])
        # Separate from the contact-sensor slots below: ``foot_z`` is indexed by the
        # articulation's body order, ``is_contact`` by the sensor's. They resolve the same
        # regex but are different lists, and mixing them is exactly the class of error
        # this file keeps collecting.
        self._front_link_slots = [i for i, n in enumerate(_foot_link_names) if n.startswith("F")]
        self._rear_link_slots = [i for i, n in enumerate(_foot_link_names) if n.startswith("R")]
        # Front/rear slots MUST be derived from the contact sensor's own ordering, because
        # that is what ``is_contact`` is indexed by in _update_command -- not from the
        # articulation's body order, which is a different list that merely happens to
        # resolve the same regex. Getting this backwards would silently report "landed
        # front-first" for exactly the behaviour being investigated, which is the failure
        # mode プロジェクト_報酬項の名前と実測対象のズレ7例.md exists to prevent.
        _sensor_names = self.contact_sensor.body_names
        _feet_ids = self.cfg.contact_sensor_cfg.body_ids
        _foot_names = [_sensor_names[i] for i in _feet_ids] if not isinstance(_feet_ids, slice) else list(_sensor_names)
        # Go2 body names are FL_foot / FR_foot / RL_foot / RR_foot.
        self._front_foot_slots = [i for i, n in enumerate(_foot_names) if n.startswith("F")]
        self._rear_foot_slots = [i for i, n in enumerate(_foot_names) if n.startswith("R")]
        if len(self._front_foot_slots) != 2 or len(self._rear_foot_slots) != 2:
            raise ValueError(
                f"expected 2 front and 2 rear feet in the contact sensor, got "
                f"{_foot_names} -> front={self._front_foot_slots} rear={self._rear_foot_slots}"
            )
        self.peak_foot_clearance = torch.zeros(self.num_envs, device=self.device)
        self.peak_pitch_rate = torch.zeros(self.num_envs, device=self.device)
        # Exposed for mdp.jump_roll_rate (Loop 30). Rewards are evaluated before the
        # command manager updates, so a reward reading this sees the PREVIOUS step's value
        # -- one step of a ~25-step flight. It errs by charging the touchdown step as if it
        # were still airborne, which is bounded by the term's own clamp.
        self.airborne_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 2026-09-09 (Loop 30 Step 0). Measurement only -- no reward reads any of these yet.
        #
        # 43 unaided jumps traced in mujoco on model_26600 (23 landed, 20 did not) say the
        # landing is decided by the angular momentum the take-off hands over, and that two
        # components of it separate success from failure about equally well: the signed
        # pitch rate during the flight (81% of attempts classified by its sign alone) and
        # the roll rate (79%). Flight time and jump height separate nothing (0.03 and 0.12
        # standardised difference) -- nothing is decided in the air.
        #
        # The cause side is a left-right asymmetry in the take-off pose. At the lift-off
        # instant the rear calves sit 1.017 rad (57 deg) apart, and that gap correlates
        # with the flight roll rate at r = -0.86; the front hips are 0.80 rad off mirror
        # symmetry and correlate at r = +0.77. Neither has ever been measured or rewarded.
        #
        # These four exist to answer one question before any reward is written: does the
        # same asymmetry exist in Isaac? The correlations above were all measured in mujoco
        # under the Loop 29 actuator DR, and a reward can only act on the Isaac side. If
        # ``liftoff_rear_calf_asym`` comes back under 0.2 rad here, the asymmetry is a
        # mujoco-side artefact and jump_takeoff_symmetry must not be built.
        #
        # Mirror convention comes from default_joint_pos, not from a naming rule:
        # hips are [0.1, -0.1, 0.1, -0.1] so a symmetric pose has q_L = -q_R (SUM is zero),
        # while thighs and calves are equal left/right (DIFFERENCE is zero).
        _jn = self.robot.joint_names
        def _slot(name: str) -> int:
            hits = [i for i, n in enumerate(_jn) if n.startswith(name)]
            if len(hits) != 1:
                raise ValueError(f"jump_command: joint {name!r} matched {hits} in {_jn}")
            return hits[0]
        self._rl_calf, self._rr_calf = _slot("RL_calf"), _slot("RR_calf")
        self._fl_hip, self._fr_hip = _slot("FL_hip"), _slot("FR_hip")
        self.liftoff_rear_calf_asym = torch.zeros(self.num_envs, device=self.device)
        self.liftoff_front_hip_asym = torch.zeros(self.num_envs, device=self.device)
        # Means over the flight, not peaks. Loop 29 named the peak pitch rate as "the one
        # number that has to move" and it did not move, while the landing rate went from
        # 0/8 to 23/43 -- the peak is the push-off transient, and what decides the landing
        # is the steady rotation that follows it. That mistake is the 11th entry in the
        # name-vs-measured list, so the signed mean is recorded separately here rather than
        # inferred from peak_pitch_rate.
        self._flight_roll_sum = torch.zeros(self.num_envs, device=self.device)
        self._flight_pitch_sum = torch.zeros(self.num_envs, device=self.device)
        self._flight_steps = torch.zeros(self.num_envs, device=self.device)
        # Signed, from projected gravity's x component at the touchdown instant. Measured
        # on the Loop 12 policy: -0.173 while landed_rear_first read 1.00, so on this
        # robot NEGATIVE is the nose-up attitude that puts the rear feet down first. The
        # sign is stated from the measurement rather than from a rotation convention.
        self.landing_pitch = torch.zeros(self.num_envs, device=self.device)
        self.landed_rear_first = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 2026-09-09 (Loop 33 Step 0). Measurement only -- no reward reads these.
        #
        # tanaka, after jumping model_27200: "the landing rate is very low, and even when
        # it lands it is a case of just about staying up rather than landing lightly. You
        # could not put this on the real robot. Go2 pushes off with BOTH rear legs at once
        # and lands on BOTH at once, and that is braking it. Most quadrupeds taking a
        # running jump plant the FRONT legs first to set the body angle at the target, push
        # off with the REAR legs after that, then land front feet first with the rear feet
        # following immediately -- staggering the two touchdowns spreads the impact."
        #
        # The project has worked on landing ORDER (jump_landing_order, Loop 19, drove
        # rear_first_fraction 0.580 -> 0.035) and on the front/rear foot HEIGHT difference
        # at touchdown (jump_landing_gear, Loop 14/20/21). It has never measured the thing
        # that spreads the impact, which is a TIME lag, nor anything at all about the leg
        # sequence during the push-off. landed_rear_first is a single boolean sampled at
        # one instant; it cannot tell a 10 ms stagger from a simultaneous slam.
        #
        # takeoff_lag  = (last rear contact) - (last front contact) before the flight.
        #                Positive = the rear feet stayed down longer, i.e. the front lifted
        #                first and the rear pushed last, which is the pattern described.
        # landing_lag  = (first rear contact) - (first front contact) after the flight.
        #                Positive = front feet touched first, rear followed.
        # landing_impact = peak summed vertical foot force after touchdown. This is the
        #                "landed lightly vs. survived it" number, and the reason the
        #                existing success_rate cannot express tanaka's complaint: an
        #                episode counts as a success on feet-down plus upright no matter
        #                how hard it arrived.
        self.takeoff_lag = torch.zeros(self.num_envs, device=self.device)
        self.landing_lag = torch.zeros(self.num_envs, device=self.device)
        self.landing_impact = torch.zeros(self.num_envs, device=self.device)
        self._t_front_off = torch.zeros(self.num_envs, device=self.device)
        self._t_rear_off = torch.zeros(self.num_envs, device=self.device)
        self._t_front_on = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._t_rear_on = torch.full((self.num_envs,), float("nan"), device=self.device)
        # 2026-09-09 (Loop 34 Step 0). Three rewards have now failed to move the 846 N
        # landing impact: take-off symmetry (Loop 30, targeted a quantity absent in Isaac),
        # landing stagger (correlated +0.20 with the impact -- the wrong sign on this
        # machine), and a direct penalty on the impact itself (946 iterations at weight
        # -25, impact flat, height -11%, falls 0% -> 2.4%).
        #
        # So stop designing and ask what actually sets the number. A landing absorbs a
        # fixed momentum, m*dv, decided by how fast the robot is falling; the PEAK force is
        # that impulse divided by the time taken to absorb it. Only two things can lower
        # it: arrive slower (which means jumping lower, and tanaka has ruled that out), or
        # take longer to stop, which needs leg travel.
        #
        # These three say which. If the knees are barely compressing, there is room and the
        # lever is the landing crouch. If they are already using their range, the impact is
        # a property of the jump and no reward will separate them.
        self.touchdown_vz = torch.zeros(self.num_envs, device=self.device)
        self.absorb_time = torch.zeros(self.num_envs, device=self.device)
        self.knee_travel = torch.zeros(self.num_envs, device=self.device)
        self._td_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._td_calf = torch.zeros(self.num_envs, device=self.device)
        self._min_calf = torch.zeros(self.num_envs, device=self.device)
        self.landing_max_tilt = torch.zeros(self.num_envs, device=self.device)
        # 2026-09-09 (Loop 34), tanaka: "it sometimes lands on its knees and elbows -- add
        # a penalty for that." A penalty already exists (undesired_contacts, weight -1, on
        # Head/hip/thigh/calf) but has never been checked for whether it fires at a landing
        # or what it pays. This measures the thing before the weight is touched.
        _cn = self.contact_sensor.body_names
        self._knee_slots = [i for i, n in enumerate(_cn) if ("_calf" in n or "_thigh" in n)]
        if not self._knee_slots:
            raise ValueError(f"jump_command: no calf/thigh bodies in contact sensor: {_cn}")
        self.landing_knee_contact = torch.zeros(self.num_envs, device=self.device)
        # Per-step flag for mdp.jump_leg_contact (the latched one above is per-jump).
        self.knee_contact_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._landing_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 2026-09-06 (Loop 14). Loop 13 removed the nose-up attitude (landing pitch
        # -0.173 -> -0.035) and rear_first_fraction did not move off 1.00, so the landing
        # order is not the body's attitude -- it is where the legs are. This is that
        # number: front foot height minus rear foot height at the touchdown instant,
        # positive meaning the front feet are still tucked up when the rear ones arrive,
        # which is tanaka's "the front legs still think they are flying".
        self.landing_front_rear_delta = torch.zeros(self.num_envs, device=self.device)
        # Why the jump window closed. Loop 12 and Loop 13 both lost success_rate at the
        # same capability level (clearance ~0.25 m, v_z ~1.9) rather than after the same
        # number of iterations, so the question is which clause of ``landed`` stops being
        # satisfiable -- there has never been a measurement that says.
        self.window_timed_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 2026-09-06 (Loop 15): how long the jump window actually stays open. This is the
        # number deploy_hold_time_s has to equal -- State_RLBase.cpp cannot detect a
        # landing and holds jump_command at 1 for a fixed duration, so any drift between
        # the two is a distribution mismatch at exactly the moment of landing. It was last
        # measured in v6b (0.68 s) and has never been re-derived since.
        self.window_length = torch.zeros(self.num_envs, device=self.device)
        # 2026-09-06 (Loop 17). Time since the latched take-off, used to bound how long the
        # dense height terms pay out. Loop 16's regression was a reward exploit: those
        # terms score quantities that are LATCHED at the take-off (liftoff_vel_z,
        # max_height_gain) but were paid every step the window stayed open, so simply not
        # landing multiplied the income. Measured across the regression, the window went
        # 0.81 -> 2.09 s and jump_takeoff_apex's income went 0.75 -> 1.93 while every
        # landing term went to ~0 and bad_orientation rose 0.006 -> 0.10.
        self.time_since_liftoff = torch.zeros(self.num_envs, device=self.device)
        # 2026-09-06 (Loop 19). v_z has stopped moving: 2.62-2.65 across the whole of
        # Loops 17 and 18 while jump_takeoff_apex still has gradient (0.78 of its target).
        # Loop 16's note said that if v_z stalled with the reward still pulling, the
        # constraint would be mechanical -- this measures whether that is true. Peak
        # applied joint torque during the take-off, as a fraction of the actuator's own
        # effort limit: at 1.0 the machine is saturated and no reward can buy more height.
        self.peak_torque_frac = torch.zeros(self.num_envs, device=self.device)
        self.upright_at_close = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.feet_down_at_close = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.metrics["jumping_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["airborne_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["assist_scale"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        # The three numbers the previous run was missing entirely.
        self.metrics["max_height_gain"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["airborne_time"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["jump_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["real_jump_fraction"] = torch.zeros(self.num_envs, device=self.device)
        # The assist-free truth. These, not the pooled numbers above, are what the
        # deployed policy will actually reproduce in mujoco / on hardware.
        self.metrics["unaided_max_height_gain"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_real_jump_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_jump_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_airborne_time"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_liftoff_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_liftoff_vel_z"] = torch.zeros(self.num_envs, device=self.device)
        # 2026-09-04 (Loop 10): the two quantities the Loop 9 post-mortem showed are the
        # ones that actually bind. ``unaided_takeoff_energy`` is v_x^2 + v_z^2, i.e. twice
        # the specific kinetic energy carried into the flight -- measured across v8 it
        # explains the distance far better than the v_x/v_z split does (corr(v_x, v_z) was
        # +0.53, so the two rise and fall together rather than trading off).
        # ``unaided_liftoff_index`` is which flight phase of the window was the take-off.
        self.metrics["unaided_takeoff_energy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_liftoff_index"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_trigger_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_stance_entry_speed"] = torch.zeros(self.num_envs, device=self.device)
        # Loop 13 additions -- see the state block above for why each exists.
        self.metrics["unaided_peak_foot_clearance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_peak_pitch_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_landing_pitch"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_rear_first_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_front_rear_delta"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_timeout_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_upright_at_close"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_feet_down_at_close"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_window_length"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_peak_torque_frac"] = torch.zeros(self.num_envs, device=self.device)
        # Loop 30 Step 0 (2026-09-09), measurement only -- see the state block above.
        self.metrics["unaided_liftoff_rear_calf_asym"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_liftoff_front_hip_asym"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_flight_roll_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_flight_pitch_rate"] = torch.zeros(self.num_envs, device=self.device)
        # Loop 31: the value of the approach gate that scales jump_takeoff_apex, so the
        # term's own multiplier is visible rather than inferred from trigger_speed.
        self.metrics["unaided_approach_gate"] = torch.zeros(self.num_envs, device=self.device)
        # Loop 33 Step 0 (2026-09-09), measurement only -- see the state block above.
        self.metrics["unaided_takeoff_lag_ms"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_landing_lag_ms"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_landing_impact_N"] = torch.zeros(self.num_envs, device=self.device)
        # 2026-09-09 (Loop 33, second pass). Raising jump_landing_stagger's weight 6 -> 30
        # took its income from +0.036 to +0.28 (the largest landing term) and moved the lag
        # -6.9 -> +4.3 ms, and the landing impact did not change by 1 N: 853 N throughout,
        # at lags from -14.7 to +4.3 ms. Before spending a loop on a larger target lag, the
        # question is whether the impact depends on the lag AT ALL. These three answer it
        # per-env instead of through the means, which cannot show a relationship.
        self.metrics["unaided_lag_impact_corr"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_impact_front_first"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_impact_rear_first"] = torch.zeros(self.num_envs, device=self.device)
        # Loop 34 Step 0 (2026-09-09), measurement only -- see the state block above.
        self.metrics["unaided_touchdown_vz"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_absorb_ms"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_knee_travel_rad"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_landing_max_tilt"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unaided_landing_knee_contact"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Shape (num_envs, 1): 1.0 while in jump mode, 0.0 while running."""
        return self.jump_command

    @property
    def enabled(self) -> torch.Tensor:
        """Shape (num_envs,) bool: True while in jump mode."""
        return self.jump_command[:, 0] > 0.0

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # Always zero on an actual episode reset, THEN let the base class's
        # reset() -> _resample() -> _resample_command() coin-flip decide whether the
        # fresh episode starts in jump mode.
        if env_ids is None:
            env_ids = slice(None)
        self.jump_command[env_ids, 0] = 0.0
        self.time_since_trigger[env_ids] = 0.0
        self.has_been_airborne[env_ids] = False
        self._prev_all_airborne[env_ids] = False
        self.success[env_ids] = False
        self.episode_real_jump[env_ids] = False
        self.max_height_gain[env_ids] = 0.0
        self.airborne_time[env_ids] = 0.0
        self.jump_distance[env_ids] = 0.0
        self.last_jump_distance[env_ids] = 0.0
        self.real_jump[env_ids] = False
        self._liftoff_recorded[env_ids] = False
        self.liftoff_count[env_ids] = 0
        self.touched_down_once[env_ids] = False
        self.time_since_touchdown[env_ids] = 0.0
        self.liftoff_vel_x[env_ids] = 0.0
        self.liftoff_vel_z[env_ids] = 0.0
        self.liftoff_index[env_ids] = 0.0
        self.trigger_speed[env_ids] = 0.0
        self.takeoff_stance_entry_speed[env_ids] = 0.0
        self._last_touchdown_speed[env_ids] = 0.0
        self.peak_foot_clearance[env_ids] = 0.0
        self.peak_pitch_rate[env_ids] = 0.0
        self.airborne_now[env_ids] = False
        self.liftoff_rear_calf_asym[env_ids] = 0.0
        self.liftoff_front_hip_asym[env_ids] = 0.0
        self._flight_roll_sum[env_ids] = 0.0
        self._flight_pitch_sum[env_ids] = 0.0
        self._flight_steps[env_ids] = 0.0
        self.landing_pitch[env_ids] = 0.0
        self.landed_rear_first[env_ids] = False
        self.takeoff_lag[env_ids] = 0.0
        self.landing_lag[env_ids] = 0.0
        self.landing_impact[env_ids] = 0.0
        self._t_front_off[env_ids] = 0.0
        self._t_rear_off[env_ids] = 0.0
        self._t_front_on[env_ids] = float("nan")
        self._t_rear_on[env_ids] = float("nan")
        self.touchdown_vz[env_ids] = 0.0
        self.absorb_time[env_ids] = 0.0
        self.knee_travel[env_ids] = 0.0
        self._td_latched[env_ids] = False
        self.landing_max_tilt[env_ids] = 0.0
        self.landing_knee_contact[env_ids] = 0.0
        self._landing_recorded[env_ids] = False
        self.landing_front_rear_delta[env_ids] = 0.0
        self.window_timed_out[env_ids] = False
        self.window_length[env_ids] = 0.0
        self.time_since_liftoff[env_ids] = 0.0
        self.peak_torque_frac[env_ids] = 0.0
        self.upright_at_close[env_ids] = False
        self.feet_down_at_close[env_ids] = False
        return super().reset(env_ids)

    def _approach_gate_bounds(self) -> tuple[float, float]:
        """The (lo, hi) actually used by ``mdp.jump_takeoff_apex``'s approach gate.

        2026-09-14. This metric used to hardcode Go2's (2.0, 2.5). The Anaguma port moved
        the reward's gate to (0.5, 1.5) for its slow-approach phase and the metric went on
        reporting the old gate -- i.e. the number named ``approach_gate`` would no longer
        have been the gate the reward applies. That is the failure this project has now
        made eight times (docs/longjump_height.md, 「名前と実測対象のズレ」), so the metric
        reads the reward term's own params instead of keeping a second copy of them.
        """
        try:
            params = self._env.reward_manager.get_term_cfg("jump_takeoff_apex").params
            return float(params.get("approach_lo", 2.0)), float(params.get("approach_hi", 2.5))
        except Exception:
            # No such reward term (a config that dropped it), or the manager is not built
            # yet. The gate the reward applies is then Go2's default.
            return 2.0, 2.5

    def _update_metrics(self):
        self.metrics["jumping_fraction"][:] = self.jump_command[:, 0]
        self.metrics["airborne_fraction"][:] = self.has_been_airborne.float()
        self.metrics["assist_scale"][:] = self.assist_scale
        self.metrics["success_rate"][:] = self.curriculum_success_rate
        self.metrics["max_height_gain"][:] = self.max_height_gain
        self.metrics["airborne_time"][:] = self.airborne_time
        self.metrics["jump_distance"][:] = self.last_jump_distance
        self.metrics["real_jump_fraction"][:] = self.real_jump.float()
        # Assist-free subset, broadcast as a scalar so the logged mean is the holdout
        # mean rather than a mean over all envs.
        holdout = self.assist_holdout
        if bool(holdout.any()):
            self.metrics["unaided_max_height_gain"][:] = self.max_height_gain[holdout].mean()
            self.metrics["unaided_real_jump_fraction"][:] = self.real_jump[holdout].float().mean()
            self.metrics["unaided_jump_distance"][:] = self.last_jump_distance[holdout].mean()
            self.metrics["unaided_airborne_time"][:] = self.airborne_time[holdout].mean()
            # Averaged over the holdout envs that actually produced a real jump -- a
            # mean including the non-jumping envs' zeros would say nothing about takeoff.
            scored = holdout & self.real_jump
            if bool(scored.any()):
                self.metrics["unaided_liftoff_vel_x"][:] = self.liftoff_vel_x[scored].mean()
                self.metrics["unaided_liftoff_vel_z"][:] = self.liftoff_vel_z[scored].mean()
                self.metrics["unaided_takeoff_energy"][:] = (
                    torch.square(self.liftoff_vel_x[scored]) + torch.square(self.liftoff_vel_z[scored])
                ).mean()
                self.metrics["unaided_liftoff_index"][:] = self.liftoff_index[scored].mean()
                # trigger -> stance entry is deceleration during the approach;
                # stance entry -> lift-off is what the take-off stance itself costs.
                self.metrics["unaided_trigger_speed"][:] = self.trigger_speed[scored].mean()
                self.metrics["unaided_stance_entry_speed"][:] = self.takeoff_stance_entry_speed[scored].mean()
                self.metrics["unaided_peak_foot_clearance"][:] = self.peak_foot_clearance[scored].mean()
                self.metrics["unaided_peak_pitch_rate"][:] = self.peak_pitch_rate[scored].mean()
                self.metrics["unaided_liftoff_rear_calf_asym"][:] = self.liftoff_rear_calf_asym[scored].mean()
                self.metrics["unaided_liftoff_front_hip_asym"][:] = self.liftoff_front_hip_asym[scored].mean()
                # Guard the divide: a scored env always has a flight, but an env whose
                # window opened this step has not accumulated one yet.
                _n = self._flight_steps[scored].clamp(min=1.0)
                self.metrics["unaided_flight_roll_rate"][:] = (self._flight_roll_sum[scored] / _n).mean()
                self.metrics["unaided_flight_pitch_rate"][:] = (self._flight_pitch_sum[scored] / _n).mean()
                _lo, _hi = self._approach_gate_bounds()
                self.metrics["unaided_approach_gate"][:] = (
                    (self.trigger_speed[scored] - _lo) / (_hi - _lo)
                ).clamp(0.0, 1.0).mean()
                self.metrics["unaided_takeoff_lag_ms"][:] = self.takeoff_lag[scored].mean() * 1000.0
                _ll = self.landing_lag[scored]
                _lm = torch.isfinite(_ll) & (_ll != 0.0)
                if _lm.any():
                    self.metrics["unaided_landing_lag_ms"][:] = _ll[_lm].mean() * 1000.0
                self.metrics["unaided_landing_impact_N"][:] = self.landing_impact[scored].mean()
                # Correlation and the two conditional means, over scored envs whose landing
                # actually recorded both sides and a non-zero impact.
                _l = self.landing_lag[scored]
                _im = self.landing_impact[scored]
                _ok = torch.isfinite(_l) & (_l != 0.0) & (_im > 1.0)
                if _ok.sum() > 20:
                    _a, _b = _l[_ok], _im[_ok]
                    _a = _a - _a.mean()
                    _b = _b - _b.mean()
                    _den = (_a.square().sum() * _b.square().sum()).sqrt()
                    if _den > 1e-9:
                        self.metrics["unaided_lag_impact_corr"][:] = (_a * _b).sum() / _den
                    _ff = _l[_ok] > 0.010
                    _rf = _l[_ok] < -0.010
                    if _ff.any():
                        self.metrics["unaided_impact_front_first"][:] = _im[_ok][_ff].mean()
                    if _rf.any():
                        self.metrics["unaided_impact_rear_first"][:] = _im[_ok][_rf].mean()
                _td = self._td_latched[scored]
                if _td.any():
                    self.metrics["unaided_touchdown_vz"][:] = self.touchdown_vz[scored][_td].mean()
                    self.metrics["unaided_absorb_ms"][:] = self.absorb_time[scored][_td].mean() * 1000.0
                    self.metrics["unaided_knee_travel_rad"][:] = self.knee_travel[scored][_td].mean()
                    self.metrics["unaided_landing_max_tilt"][:] = self.landing_max_tilt[scored][_td].mean()
                    self.metrics["unaided_landing_knee_contact"][:] = self.landing_knee_contact[scored][_td].mean()
                landed = scored & self._landing_recorded
                if bool(landed.any()):
                    self.metrics["unaided_landing_pitch"][:] = self.landing_pitch[landed].mean()
                    self.metrics["unaided_rear_first_fraction"][:] = self.landed_rear_first[landed].float().mean()
                    self.metrics["unaided_front_rear_delta"][:] = self.landing_front_rear_delta[landed].mean()
                closed = holdout & (self.window_timed_out | self.success)
                if bool(closed.any()):
                    self.metrics["unaided_timeout_fraction"][:] = self.window_timed_out[closed].float().mean()
                    self.metrics["unaided_upright_at_close"][:] = self.upright_at_close[closed].float().mean()
                    self.metrics["unaided_feet_down_at_close"][:] = self.feet_down_at_close[closed].float().mean()
                    self.metrics["unaided_window_length"][:] = self.window_length[closed].mean()
                if bool(scored.any()):
                    self.metrics["unaided_peak_torque_frac"][:] = self.peak_torque_frac[scored].mean()

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return
        # Only coin-flip envs that are NOT currently jumping.
        idle_ids = env_ids_t[self.jump_command[env_ids_t, 0] == 0.0]
        if idle_ids.numel() == 0:
            return
        trigger = torch.rand(idle_ids.numel(), device=self.device) < self.cfg.jump_probability
        triggered_ids = idle_ids[trigger]
        self.jump_command[triggered_ids, 0] = 1.0
        self.time_since_trigger[triggered_ids] = 0.0
        self.has_been_airborne[triggered_ids] = False
        # Per-jump measurement baseline.
        self.trigger_height[triggered_ids] = self.robot.data.root_pos_w[triggered_ids, 2]
        self.max_height_gain[triggered_ids] = 0.0
        self.peak_foot_clearance[triggered_ids] = 0.0
        self.peak_pitch_rate[triggered_ids] = 0.0
        self.airborne_now[triggered_ids] = False
        self.liftoff_rear_calf_asym[triggered_ids] = 0.0
        self.liftoff_front_hip_asym[triggered_ids] = 0.0
        self._flight_roll_sum[triggered_ids] = 0.0
        self._flight_pitch_sum[triggered_ids] = 0.0
        self._flight_steps[triggered_ids] = 0.0
        self.landing_pitch[triggered_ids] = 0.0
        self.landed_rear_first[triggered_ids] = False
        self.takeoff_lag[triggered_ids] = 0.0
        self.landing_lag[triggered_ids] = 0.0
        self.landing_impact[triggered_ids] = 0.0
        self._t_front_off[triggered_ids] = 0.0
        self._t_rear_off[triggered_ids] = 0.0
        self._t_front_on[triggered_ids] = float("nan")
        self._t_rear_on[triggered_ids] = float("nan")
        self.touchdown_vz[triggered_ids] = 0.0
        self.absorb_time[triggered_ids] = 0.0
        self.knee_travel[triggered_ids] = 0.0
        self._td_latched[triggered_ids] = False
        self.landing_max_tilt[triggered_ids] = 0.0
        self.landing_knee_contact[triggered_ids] = 0.0
        self._landing_recorded[triggered_ids] = False
        self.landing_front_rear_delta[triggered_ids] = 0.0
        self.window_timed_out[triggered_ids] = False
        self.window_length[triggered_ids] = 0.0
        self.time_since_liftoff[triggered_ids] = 0.0
        self.peak_torque_frac[triggered_ids] = 0.0
        self.upright_at_close[triggered_ids] = False
        self.feet_down_at_close[triggered_ids] = False
        self.airborne_time[triggered_ids] = 0.0
        self.real_jump[triggered_ids] = False
        self._liftoff_recorded[triggered_ids] = False
        self.liftoff_count[triggered_ids] = 0
        self.touched_down_once[triggered_ids] = False
        self.time_since_touchdown[triggered_ids] = 0.0
        self.liftoff_vel_x[triggered_ids] = 0.0
        self.liftoff_vel_z[triggered_ids] = 0.0
        self.liftoff_index[triggered_ids] = 0.0
        approach_speed = torch.norm(self.robot.data.root_lin_vel_w[triggered_ids, :2], dim=-1)
        self.trigger_speed[triggered_ids] = approach_speed
        self.takeoff_stance_entry_speed[triggered_ids] = 0.0
        # Seeded rather than zeroed: a window that opens mid-flight has no touchdown of
        # its own yet, and the speed it entered with is the honest stand-in.
        self._last_touchdown_speed[triggered_ids] = approach_speed

    def _update_command(self):
        jumping = self.enabled
        self.time_since_trigger[jumping] += self._env.step_dt

        feet_ids = self.cfg.contact_sensor_cfg.body_ids
        is_contact = self.contact_sensor.data.current_contact_time[:, feet_ids] > 0.0
        # 2026-09-03 (v6): was ``is_contact.all(dim=-1)``, i.e. all four feet in contact
        # at the same instant. A quadruped landing out of a 3+ m/s run puts its feet down
        # one or two at a time and runs on; all-four-simultaneously is a standing pose, so
        # the landing condition almost never fired -- success_rate sat at 1.4% early in v5
        # and jump_distance (weight 20, the task objective itself) was paid essentially
        # never. Two feet is what a running touchdown actually looks like.
        enough_feet_down = is_contact.sum(dim=-1) >= self.cfg.min_feet_down_for_landing
        all_airborne = ~is_contact.any(dim=-1)

        # Track a genuine airborne phase (only counts while actually commanded to jump).
        self.has_been_airborne |= all_airborne & jumping

        # --- Loop 33 Step 0: front/rear leg sequencing and landing impact ---
        # The two sides are tracked separately rather than as "how many feet are down",
        # because the quantity in question is WHICH END and WHEN, not how many.
        _front_c = is_contact[:, self._front_foot_slots].any(dim=-1)
        _rear_c = is_contact[:, self._rear_foot_slots].any(dim=-1)
        _t = self.time_since_trigger
        # Before the flight: keep overwriting, so each side ends up holding the last
        # instant it was on the ground -- the moment that side left for the take-off.
        _pre = jumping & (~self.has_been_airborne)
        self._t_front_off = torch.where(_pre & _front_c, _t, self._t_front_off)
        self._t_rear_off = torch.where(_pre & _rear_c, _t, self._t_rear_off)
        # After the flight: latch the FIRST contact of each side, once.
        _post = jumping & self.has_been_airborne
        _new_front = _post & _front_c & torch.isnan(self._t_front_on)
        _new_rear = _post & _rear_c & torch.isnan(self._t_rear_on)
        self._t_front_on = torch.where(_new_front, _t, self._t_front_on)
        self._t_rear_on = torch.where(_new_rear, _t, self._t_rear_on)
        _both = torch.isfinite(self._t_front_on) & torch.isfinite(self._t_rear_on)
        self.takeoff_lag = torch.where(_post, self._t_rear_off - self._t_front_off, self.takeoff_lag)
        self.landing_lag = torch.where(_both, self._t_rear_on - self._t_front_on, self.landing_lag)
        # Peak summed vertical foot force once the landing has begun. Sum, not max over
        # feet: a landing that puts the same total load through two feet instead of four
        # is exactly what "spreading the impact" is supposed to avoid, and a per-foot max
        # would read the same for both.
        # Absorption: latch the state at the first contact after the flight, then run the
        # clock and track the deepest knee flexion while the trunk is still descending.
        _calf = self.robot.data.joint_pos[:, 8:12].mean(dim=1)
        _vz = self.robot.data.root_lin_vel_w[:, 2]
        # 2026-09-09, FIXED. The first version latched on the first contact after
        # ``has_been_airborne``, and measured touchdown v_z = +0.40 m/s (trunk RISING),
        # knee travel 0.133 rad (7% of range) and 878 N -- with the impulse/time estimate
        # off by 55x. It was not measuring the jump's landing at all. ``has_been_airborne``
        # is set by ANY flight phase inside the window, and at a 2.8 m/s approach every
        # running stride has one; ``unaided_liftoff_index`` has read 2.0 since Loop 10,
        # i.e. the take-off is always the window's SECOND flight. So this latched the
        # touchdown of the ordinary stride that precedes the take-off, and 878 N is a
        # running ground reaction, not a landing impact.
        #
        # Name-vs-measured #12, and the fifth caused by ``has_been_airborne`` at speed --
        # the failure this project has repeated more than any other, committed here by a
        # metric added specifically to diagnose it. The gate has to be the LATCHED
        # take-off: contact resuming after the flight that ``time_since_liftoff`` counts.
        _new_td = (
            jumping
            & self._liftoff_recorded
            & (self.time_since_liftoff > 0.0)
            & self._prev_all_airborne
            & (_front_c | _rear_c)
            & (~self._td_latched)
        )
        self.touchdown_vz[_new_td] = _vz[_new_td]
        self._td_calf[_new_td] = _calf[_new_td]
        self._min_calf[_new_td] = _calf[_new_td]
        self._td_latched |= _new_td
        # Clock from the latched touchdown, used both as the impact window and as the
        # absorption time; the descent-only version could not run at all when the latch
        # was firing mid-stride with the trunk rising.
        self.absorb_time += (self._td_latched & jumping).float() * self._env.step_dt
        _absorbing = self._td_latched & jumping & (_vz < 0.0)
        self._min_calf = torch.where(_absorbing, torch.minimum(self._min_calf, _calf), self._min_calf)
        self.knee_travel = torch.where(self._td_latched, self._td_calf - self._min_calf, self.knee_travel)
        _fz = self.contact_sensor.data.net_forces_w[:, feet_ids, 2].abs().sum(dim=-1)
        # Peak within 60 ms of the latched touchdown only. The first version took the peak
        # over 0.3 s, which at a 2.8 m/s approach is long enough for the robot to have
        # taken another running stride -- the peak was that stride's push-off.
        _impact_win = self._td_latched & (self.absorb_time <= 0.06) & (_front_c | _rear_c)
        self.landing_impact = torch.where(
            _impact_win, torch.maximum(self.landing_impact, _fz), self.landing_impact
        )
        # Posture disturbance after the landing: "just about stayed up" should show here
        # even when the force does not.
        _tilt = torch.acos((-self.robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)).abs()
        self.landing_max_tilt = torch.where(
            self._td_latched & jumping, torch.maximum(self.landing_max_tilt, _tilt), self.landing_max_tilt
        )
        # Any thigh or calf link touching the ground from the latched take-off onwards.
        # Latched, so it reports "did this jump land on a knee", not an instantaneous state.
        _knee_now = (
            self.contact_sensor.data.net_forces_w[:, self._knee_slots, :].norm(dim=-1) > 1.0
        ).any(dim=-1)
        self.knee_contact_now[:] = _knee_now & self._liftoff_recorded & jumping
        self.landing_knee_contact = torch.where(
            self._liftoff_recorded & jumping & _knee_now,
            torch.ones_like(self.landing_knee_contact),
            self.landing_knee_contact,
        )

        # --- measurement ---
        root_pos_w = self.robot.data.root_pos_w
        height_gain = root_pos_w[:, 2] - self.trigger_height
        self.max_height_gain = torch.where(
            jumping, torch.maximum(self.max_height_gain, height_gain), self.max_height_gain
        )
        airborne_now = all_airborne & jumping
        self.airborne_now[:] = airborne_now
        self.airborne_time += airborne_now.float() * self._env.step_dt

        # --- 2026-09-06 (Loop 13) attitude / clearance measurement -------------------
        # Ground clearance of the LOWEST foot: what a person watching calls the height of
        # the jump. The terrain is a plane at z = 0 (see RobotEnvCfgLongJumpBase), so the
        # world z of the foot body IS its clearance, up to the foot's collision radius --
        # a constant offset, so changes in this number are still the changes that matter.
        foot_z = self.robot.data.body_pos_w[:, self._foot_link_ids, 2]
        lowest_foot = foot_z.min(dim=1).values
        self.peak_foot_clearance = torch.where(
            jumping, torch.maximum(self.peak_foot_clearance, lowest_foot), self.peak_foot_clearance
        )
        # Pitch rate, tracked only while actually in the air -- on the ground the legs
        # produce plenty of body rotation that has nothing to do with the flight attitude.
        pitch_rate = self.robot.data.root_ang_vel_b[:, 1].abs()
        self.peak_pitch_rate = torch.where(
            airborne_now, torch.maximum(self.peak_pitch_rate, pitch_rate), self.peak_pitch_rate
        )
        # Flight-phase means. Accumulated only while all four feet are off the ground, so
        # the push-off transient (which is what peak_pitch_rate captures) is excluded.
        # Roll is magnitude -- the failures roll either way; pitch keeps its sign, because
        # the mujoco traces separate on the sign and not on the magnitude.
        _fly = airborne_now.float()
        self._flight_roll_sum += self.robot.data.root_ang_vel_b[:, 0].abs() * _fly
        self._flight_pitch_sum += self.robot.data.root_ang_vel_b[:, 1] * _fly
        self._flight_steps += _fly
        # At the touchdown instant, record the attitude and which end arrived first.
        # ``_prev_all_airborne`` still holds the previous step here (it is refreshed at the
        # end of this method), so this is the rising edge of contact after a flight phase.
        # Gated on ``real_jump`` for the same reason touched_down_once is (see below): at a
        # 2 m/s approach every running stride has a flight phase, so the first contact
        # inside the jump window is usually an ordinary stride landing, not the jump's.
        # ``real_jump`` is this step's value from the previous update, which is what we
        # want -- it latches during the flight, before the touchdown being measured.
        touchdown_edge = (
            jumping
            & self.real_jump
            & self._prev_all_airborne
            & is_contact.any(dim=-1)
            & (~self._landing_recorded)
        )
        if bool(touchdown_edge.any()):
            rear_down = is_contact[:, self._rear_foot_slots].any(dim=-1)
            front_down = is_contact[:, self._front_foot_slots].any(dim=-1)
            self.landing_pitch = torch.where(
                touchdown_edge, self.robot.data.projected_gravity_b[:, 0], self.landing_pitch
            )
            self.landed_rear_first = torch.where(
                touchdown_edge, rear_down & (~front_down), self.landed_rear_first
            )
            front_z = foot_z[:, self._front_link_slots].mean(dim=1)
            rear_z = foot_z[:, self._rear_link_slots].mean(dim=1)
            self.landing_front_rear_delta = torch.where(
                touchdown_edge, front_z - rear_z, self.landing_front_rear_delta
            )
            self._landing_recorded |= touchdown_edge
        # Position at the instant the last foot FIRST left the ground in this window.
        # Latch-once (``_liftoff_recorded``): re-latching on every flight phase is what
        # made v4's jump_distance read 0.06-0.15 m -- it measured only the final hop
        # before touchdown instead of the whole jump.
        liftoff_edge = airborne_now & (~self._prev_all_airborne)
        # 2026-09-03: latch the STRONGEST liftoff of the window, not the first one.
        #
        # "First" was wrong for the same reason has_been_airborne is wrong: at a 3.3 m/s
        # approach every running stride has a flight phase, so the first all-four-feet-off
        # event in a jump window is an ordinary stride that happens to precede the
        # push-off. Measured directly, the latched liftoff had a *downward* vertical
        # velocity of -0.33 m/s -- the descending half of a stride, not a takeoff. Every
        # jump_distance number reported so far was therefore measured from a running
        # stride rather than from the jump, which is why v6b's taller jump appeared to
        # travel less far than v5's.
        #
        # Taking the liftoff with the highest upward velocity needs no threshold to tune
        # and is what "takeoff" physically means: the most energetic push-off in the
        # window is the jump.
        root_vel_w = self.robot.data.root_lin_vel_w
        # Horizontal speed entering each stance, so the strongest-liftoff latch below can
        # carry over the speed the take-off stance actually started from.
        touchdown_edge = (~all_airborne) & self._prev_all_airborne
        self._last_touchdown_speed[touchdown_edge] = torch.norm(root_vel_w[touchdown_edge, :2], dim=-1)
        stronger = liftoff_edge & (
            (~self._liftoff_recorded) | (root_vel_w[:, 2] > self.liftoff_vel_z)
        )
        self._liftoff_pos_xy[stronger] = root_pos_w[stronger, :2]
        self.liftoff_vel_x[stronger] = torch.norm(root_vel_w[stronger, :2], dim=-1)
        self.liftoff_vel_z[stronger] = root_vel_w[stronger, 2]
        self.takeoff_stance_entry_speed[stronger] = self._last_touchdown_speed[stronger]
        # Take-off pose asymmetry, latched with the same "strongest lift-off" rule as the
        # velocities above so it describes the take-off the rewards are scoring.
        _q = self.robot.data.joint_pos
        self.liftoff_rear_calf_asym[stronger] = (
            _q[stronger, self._rl_calf] - _q[stronger, self._rr_calf]
        ).abs()
        self.liftoff_front_hip_asym[stronger] = (
            _q[stronger, self._fl_hip] + _q[stronger, self._fr_hip]
        ).abs()
        # liftoff_count is incremented below, so this edge is the (count + 1)-th.
        # Read by mdp.jump_repeat_liftoff, which charges only for flights after this one.
        self.liftoff_index[stronger] = self.liftoff_count[stronger].float() + 1.0
        # Restarts whenever a stronger take-off is latched, so the payout clock always
        # tracks the take-off the reward is actually scoring.
        # Actuator headroom during the take-off (Loop 19 diagnostic). Charged only before
        # the flight -- once airborne the legs are unloaded and the number means nothing.
        # Raw N.m, not a fraction of joint_effort_limits: that array is the SIMULATION
        # limit, which IsaacLab leaves very large while the real ceiling is enforced by the
        # actuator model, so normalising by it read 0.0000 in the smoke. The Go2's knee
        # limit is 45.43 N.m (corrected in Loop 2 from the URDF), which is the number to
        # compare this against.
        torque_frac = self.robot.data.applied_torque.abs().max(dim=1).values
        on_ground = jumping & (~all_airborne)
        self.peak_torque_frac = torch.where(
            on_ground, torch.maximum(self.peak_torque_frac, torque_frac), self.peak_torque_frac
        )
        self.time_since_liftoff += jumping.float() * self._env.step_dt
        self.time_since_liftoff[stronger] = 0.0
        self._liftoff_recorded |= liftoff_edge
        # Every flight phase in the window, so the 2nd and later ones can be charged for.
        self.liftoff_count += liftoff_edge.long()
        # First contact after the first flight: from here on the legs should compress to
        # absorb, so jump_trunk_clearance stops applying.
        # 2026-09-03 (v6b): gated on ``real_jump``, NOT ``has_been_airborne``. At a 3.3 m/s
        # approach the running gait's suspension phase sets has_been_airborne within a few
        # steps of the trigger, and the next ordinary footfall then set this flag -- so
        # "first touchdown" was landing on the first running stride, roughly 0.1 s after
        # the trigger and long before any jump. Measured: touched_down_once reached 0.87
        # while the 0.3 s landing window it feeds was satisfied 0.0000 of the time,
        # because the window had already expired by the real landing. real_jump (0.20 s
        # airborne AND 0.08 m of rise) is the gate a running stride cannot pass.
        self.touched_down_once |= self.real_jump & is_contact.any(dim=-1) & jumping
        self.time_since_touchdown += self.touched_down_once.float() * self._env.step_dt
        # "Real jump" gate: long enough in the air AND high enough to be more than the
        # suspension phase of a fast gait (which is ~0.05-0.10 s with ~no rise).
        self.real_jump |= (self.airborne_time >= self.cfg.min_airborne_s) & (
            self.max_height_gain >= self.cfg.min_height_gain_m
        )

        upright = torch.acos(-self.robot.data.projected_gravity_b[:, 2]).abs() < self.cfg.landed_upright_angle
        min_air_time_elapsed = self.time_since_trigger > self.cfg.min_air_time_s

        # A real landing: left the ground, came back down on all four feet, upright.
        landed = jumping & self.has_been_airborne & enough_feet_down & upright & min_air_time_elapsed
        # Hard timeout, pose-independent -- closes the "stay airborne-flagged forever
        # while lying on the ground" exploit (fix 2 in the class docstring).
        timed_out = jumping & (self.time_since_trigger > self.cfg.max_jump_duration_s)

        # Horizontal distance covered while airborne, paid out once at the landing of a
        # jump that cleared the real-jump gate. This is the task objective itself
        # (distance = 2*v_x*v_z/g), unlike the horizontal-lift-off-speed target it
        # replaces, which asked the robot to slow to 1.5 m/s while the velocity command
        # was pushing 4.4 m/s. See プロジェクト_StageB跳躍が小さい原因分析.md.
        self.jump_distance[:] = 0.0
        scored = landed & self.real_jump
        if torch.any(scored):
            self.jump_distance[scored] = torch.norm(
                root_pos_w[scored, :2] - self._liftoff_pos_xy[scored], dim=-1
            )
            self.last_jump_distance[scored] = self.jump_distance[scored]

        # Episode-level success for the assist curriculum: now requires a REAL jump that
        # landed upright. Previously any landing counted, which a fast run satisfies --
        # so the assist was being withdrawn for running well, not for jumping.
        self.success |= scored
        # 2026-09-03 (v5): episode-level "did a real jump happen at all", regardless of
        # whether the landing was clean enough to score. This is what jump_assist_decay
        # now gates on. ``success`` proved unusable as a curriculum signal: it additionally
        # requires all four feet down AND upright AND the window not having timed out, and
        # once airborne time grew to ~0.6 s that combination stopped fitting inside
        # max_jump_duration_s. Across the whole v4 run real_jump_fraction sat at ~90%
        # while success_rate sat at ~15%, so the assist held at 1.00 for 2900 of 3000
        # iterations and the policy never had to jump for itself.
        self.episode_real_jump |= self.real_jump

        # Loop 14 diagnostics: latch WHY the window is closing, before it is cleared.
        closing = jumping & (landed | timed_out)
        if bool(closing.any()):
            self.window_timed_out = torch.where(closing, timed_out & (~landed), self.window_timed_out)
            self.upright_at_close = torch.where(closing, upright, self.upright_at_close)
            self.feet_down_at_close = torch.where(closing, enough_feet_down, self.feet_down_at_close)
            self.window_length = torch.where(closing, self.time_since_trigger, self.window_length)
        self.jump_command[landed | timed_out, 0] = 0.0
        self.has_been_airborne[landed | timed_out] = False

        self._prev_all_airborne = all_airborne.clone()
        self._apply_assistance()

    def _apply_assistance(self):
        """EFGCL external-force assist: crouch pulse, then a ramped vertical launch.

        Ported from tak's ``JumpCommand._apply_assistance`` (tag ``jump-demo``), with the
        launch force sized for this task's own target: an apex of
        ``cfg.assist_apex_height_m`` above the take-off height, i.e. the average force
        needed to impart v0 = sqrt(2*g*h) over ``cfg.assist_duration_s``. The robot's own
        forward running speed supplies the horizontal component, so no horizontal assist
        is applied -- keeping the assist orthogonal to the quantity
        mdp.jump_sparse_reward actually scores (horizontal lift-off speed).
        """
        if self.assist_scale <= 0.0:
            return

        elapsed = self.time_since_trigger
        triggered = self.enabled
        forces = torch.zeros(self.num_envs, len(self._assist_body_ids), 3, device=self.device)
        # Per-env magnitude: the holdout group is permanently unaided, so its metrics
        # report what the policy does by itself (see __init__).
        scale = torch.full((self.num_envs,), self.assist_scale, device=self.device)
        scale[self.assist_holdout] = 0.0

        # -- crouch pulse: brief downward shove, triangular 0->peak->0 envelope so there
        #    is no force discontinuity at either end.
        crouch_duration = self.cfg.crouch_assist_duration_s
        if crouch_duration > 0.0 and self.cfg.crouch_assist_force > 0.0:
            crouch_active = triggered & (elapsed >= 0.0) & (elapsed < crouch_duration)
            if torch.any(crouch_active):
                half = crouch_duration / 2.0
                envelope = torch.minimum(elapsed / half, (crouch_duration - elapsed) / half).clamp(0.0, 1.0)
                per_body = self.cfg.crouch_assist_force * scale * envelope / len(self._assist_body_ids)
                for body_index in range(len(self._assist_body_ids)):
                    forces[crouch_active, body_index, 2] = -per_body[crouch_active]

        # -- launch force: begins exactly as the crouch pulse ends, ramped in over
        #    assist_ramp_s so both sides of the handoff sit at ~0 force.
        delay = self.cfg.assist_delay_s
        ramp = self.cfg.assist_ramp_s
        launch_active = triggered & (elapsed >= delay) & (elapsed < delay + ramp + self.cfg.assist_duration_s)
        if torch.any(launch_active):
            if ramp > 0.0:
                ramp_progress = ((elapsed - delay) / ramp).clamp(0.0, 1.0)
            else:
                ramp_progress = torch.ones_like(elapsed)
            initial_velocity = (2.0 * self.cfg.gravity * max(self.cfg.assist_apex_height_m, 0.0)) ** 0.5
            # 2026-09-05 (Loop 12): divide by the duration the force is ACTUALLY applied
            # for, not by assist_duration_s alone. launch_active runs over
            # [delay, delay + ramp + duration), and the ramp is a 0->1 linear rise, so the
            # delivered impulse is F * (ramp/2 + duration) = F * 0.16 s against the F *
            # 0.10 s this sizing assumed -- 1.6x the intended dv, i.e. 2.56x the intended
            # apex. Measured in the Loop 12 smoke: assist_apex_height_m = 0.20 produced a
            # max_height_gain of 0.40-0.61 m (predicted 0.512 m), well past the reward's
            # 0.28 m target, and flipped the robot on 35-41% of episodes.
            #
            # This also re-reads the Loop 10 decision to switch the assist off. That was
            # argued on the nominal v0 = sqrt(2*g*0.12) = 1.53 m/s being weaker than v8's
            # unaided 1.62 m/s; the force actually delivered 2.46 m/s. The teacher was
            # never weaker than the student -- the number it was compared on was the
            # configured value, not what the force delivered. Same class of error as the
            # seven collected in
            # プロジェクト_報酬項の名前と実測対象のズレ7例.md.
            applied_duration = 0.5 * max(ramp, 0.0) + self.cfg.assist_duration_s
            total_force = self._robot_mass * initial_velocity / max(applied_duration, 1e-6)
            per_body = total_force * scale / len(self._assist_body_ids)
            for body_index in range(len(self._assist_body_ids)):
                forces[launch_active, body_index, 2] = (
                    per_body[launch_active] * ramp_progress[launch_active]
                )

        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=torch.zeros_like(forces),
            body_ids=self._assist_body_ids,
            is_global=True,
        )

@configclass
class JumpCommandCfg(CommandTermCfg):
    class_type: type = JumpCommand

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot")

    jump_probability: float = MISSING
    """Probability that an idle (jump_command==0) env triggers a jump at each resample boundary."""

    min_air_time_s: float = MISSING
    """Minimum time since trigger before a landing can reset jump_command back to 0.

    Must be long enough to cover a real crouch-and-push-off (~0.2-0.4 s). At the old
    0.05 s the command cleared itself before the policy could act on it -- see the
    JumpCommand class docstring, fix (1).
    """

    max_jump_duration_s: float = MISSING
    """Hard timeout: jump_command is force-cleared this long after the trigger.

    Pose-independent on purpose -- a robot that fell over must not be able to hold the
    command at 1 forever (which previously also disabled the orientation termination).
    """

    landed_upright_angle: float = MISSING

    # 2026-09-06 (Loop 16): 0.65 -> 0.76, re-derived from unaided_window_length, which
    # Loop 15 added precisely so this number stops being a stale estimate. The window
    # settled at 0.762-0.767 s once the landing detection was fixed.
    deploy_hold_time_s: float = 0.85
    """How long the DEPLOY side holds jump_command at 1 (State_RLBase.cpp has no
    contact sensing, so it cannot detect a landing and must use a fixed duration).

    This must be the *typical* window length, not ``max_jump_duration_s`` -- that is a
    timeout the training-time window almost never reaches, because JumpCommand clears
    the command the moment a landing is detected. Setting it to the timeout leaves the
    deployed policy in jump mode for a second or more after it has already landed, which
    tanaka observed as "it lands fine now, but falls after a few steps".

    0.7 s is measured from the v6b run: jumping_fraction 0.0788 with a mean idle time of
    4.0 s / 0.5 trigger probability = 8.0 s between triggers gives a mean window of
    0.0788 * 8.0 / (1 - 0.0788) = 0.68 s.
    """

    min_feet_down_for_landing: int = 2
    """Feet in contact required to call a landing. 4 (all of them) does not work.

    2026-09-03 (v6). A quadruped touching down out of a 3+ m/s run puts its feet down
    one or two at a time; all four simultaneously is a standing pose. Requiring all four
    made the landing condition fire almost never (success_rate 1.4% early in v5), which
    in turn meant jump_distance -- weight 20, the task objective itself -- was paid
    essentially never.
    """
    """Max angle (rad) between projected gravity and the body z-axis to count as "upright"."""

    min_airborne_s: float = 0.20
    """Minimum accumulated airborne time for an attempt to count as a real jump.

    A Go2 running at 3-4.4 m/s already has a suspension phase of roughly 0.05-0.10 s,
    which is why ``has_been_airborne`` alone read 100% while mujoco showed almost no
    jump at all. 0.20 s corresponds to a ballistic rise of ~5 cm and up, so it sits
    clearly above the gait's own flight time.
    """

    min_height_gain_m: float = 0.08
    """Minimum peak rise above the trigger-instant base height for a real jump."""

    # --- EFGCL assist ---
    assist_body_names: tuple[str, ...] = ("FR_hip", "FL_hip", "RR_hip", "RL_hip")
    """Bodies the assist forces are applied to (all four legs, so the crouch is symmetric)."""

    initial_assist_scale: float = 1.0
    """Starting assist magnitude in [0, 1]. Decayed by mdp.jump_assist_decay; 0.0 disables."""

    assist_holdout_fraction: float = 0.25
    """Fraction of envs that never receive the assist, used to measure the policy alone.

    2026-09-03 (v5). Without this, every jump metric is contaminated by the external
    force: v4 logged max_height_gain 0.21 m and real_jump_fraction 90% while the
    assist -- which is sized to deliver the whole target apex by itself -- was still at
    0.94, and mujoco (no assist) showed almost no jump. The holdout group's
    ``unaided_*`` metrics are the ones to trust, because they are the only ones the
    deployed ONNX policy has to reproduce.
    """

    crouch_assist_force: float = 150.0
    """Peak downward force (N, total across assist bodies) of the crouch-load pulse."""

    crouch_assist_duration_s: float = 0.12
    """Duration of the crouch pulse. 0.0 disables it."""

    assist_delay_s: float = 0.12
    """Delay from trigger to the start of the launch ramp. Set equal to crouch_assist_duration_s."""

    assist_ramp_s: float = 0.12
    """Ramp-in time of the launch force (a hard step reliably broke training)."""

    assist_duration_s: float = 0.10
    """Duration the launch force is held at full magnitude after the ramp."""

    assist_apex_height_m: float = 0.12
    """Apex height the launch force is sized for, via v0 = sqrt(2*g*h).

    2026-09-03 (v5): 0.20 -> 0.12, and deliberately now BELOW the reward's
    ``target_height`` (0.28 m). In v4 the two were equal, which meant the teacher
    delivered exactly the apex the reward asked for: the launch impulse is
    ``m * sqrt(2*g*h)`` by construction, so at full scale the assist satisfied
    jump_height_progress on its own and the policy could collect the reward while
    contributing nothing. Sizing the assist for less than the target keeps it a
    demonstration of the *motion* while leaving the height itself as something the
    policy must supply -- from the very first iteration, not only after the assist
    decays. See プロジェクト_StageB着地失敗の原因分析.md.
    """

    gravity: float = 9.81
