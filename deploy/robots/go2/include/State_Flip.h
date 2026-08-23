// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// State_Flip: deploy-side controller for the Go2 dynamic jump / backflip /
// sideflip policies trained with `unitree_rl_lab.tasks.dynamic` (JumpCommand).
//
// The trained policy consumes two extra observation terms compared to the
// locomotion policy:
//   * jump_command : [enabled, target_height, target_pitch_turns, target_roll_turns]
//   * jump_time    : cubic time encoding measured from the command rising edge
// Both are reproduced here (see State_Flip.cpp) from the live `FlipCommand`
// state owned by the active flip state.
//
// One state class powers every motion; the concrete motion (jump / backflip /
// sideflip) is selected purely by the command targets read from config.yaml.

#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

class State_Flip : public FSMState
{
public:
    // Mirrors the timing / target bookkeeping of `JumpCommand` in
    // source/.../tasks/dynamic/mdp/commands.py, reduced to the single-robot
    // deploy case. All time-keeping is self-contained (driven by policy steps)
    // so the observation terms never depend on env->episode_length ordering.
    struct FlipCommand
    {
        // --- active motion targets (mirror the 4-D policy command vector) ---
        // Latched from the pending_* values on each trigger; read by the
        // jump_command observation term.
        float target_height = 0.0f;        // meters above nominal standing height
        float target_pitch_turns = 0.0f;   // backflip target, in turns (-1.0 == one back rotation)
        float target_roll_turns = 0.0f;    // sideflip target, in turns

        // --- pending targets (written by the keyboard/FSM thread, read on trigger) ---
        std::atomic<float> pending_height{0.0f};
        std::atomic<float> pending_pitch{0.0f};
        std::atomic<float> pending_roll{0.0f};
        std::atomic<bool> trigger_requested{false};

        // --- timing (must match the training JumpCommandCfg) ---
        float time_scale = 1.0f;           // jump_time_encoding time scale
        float command_duration_s = 0.5f;   // how long `enabled` stays high after trigger
        float trigger_delay_s = 1.0f;      // (single-shot mode) auto-trigger this long after entry
        float rearm_delay_s = 0.3f;        // (manual mode) cooldown after a motion before re-firing
        bool use_auto_trigger = true;      // single-shot: fire once automatically after trigger_delay_s
        bool manual_mode = false;          // manual: fire on demand + re-arm, target chosen per trigger

        // --- runtime state ---
        float step_dt = 0.02f;
        long step_count = 0;
        long trigger_step = -1;
        bool enabled = false;
        bool command_issued = false;

        void reset()
        {
            step_count = 0;
            trigger_step = -1;
            enabled = false;
            command_issued = false;
            trigger_requested.store(false);
            target_height = 0.0f;
            target_pitch_turns = 0.0f;
            target_roll_turns = 0.0f;
        }

        // Queue a motion (manual mode). Called from the FSM thread.
        void request(float height, float pitch_turns, float roll_turns)
        {
            pending_height.store(height);
            pending_pitch.store(pitch_turns);
            pending_roll.store(roll_turns);
            trigger_requested.store(true);
        }

        // Advance one policy step; call once *before* the observation is computed.
        void step()
        {
            step_count += 1;

            // Manual mode: once a motion has finished and the cooldown elapsed,
            // re-arm so the next keypress can fire again. Targets drop to zero in
            // the gap so the policy holds a stand between motions.
            if (manual_mode && command_issued && !enabled
                && elapsed() >= command_duration_s + rearm_delay_s)
            {
                command_issued = false;
                target_height = 0.0f;
                target_pitch_turns = 0.0f;
                target_roll_turns = 0.0f;
            }

            if (!command_issued)
            {
                bool fire = trigger_requested.exchange(false);
                if (use_auto_trigger && (step_count * step_dt) >= trigger_delay_s)
                {
                    fire = true;
                }
                if (fire)
                {
                    if (manual_mode)
                    {
                        target_height = pending_height.load();
                        target_pitch_turns = pending_pitch.load();
                        target_roll_turns = pending_roll.load();
                    }
                    enabled = true;
                    command_issued = true;
                    trigger_step = step_count;
                }
            }

            // The command expires after command_duration_s; afterwards the
            // policy sees a zero command and returns to standing.
            if (enabled && elapsed() >= command_duration_s)
            {
                enabled = false;
            }
        }

        float elapsed() const
        {
            return trigger_step >= 0 ? (step_count - trigger_step) * step_dt : 0.0f;
        }

        // Retained after the command turns off? No -- like jump_time_encoding,
        // the encoded time is gated by `enabled` so it drops back to zero.
        float time_since_trigger() const { return enabled ? elapsed() : 0.0f; }
    };

    // A keyboard-selectable motion preset (manual/Dynamic mode).
    struct MotionPreset
    {
        std::string key;
        std::string name;
        float target_height = 0.0f;
        float target_pitch_turns = 0.0f;
        float target_roll_turns = 0.0f;
    };

    State_Flip(int state_mode, std::string state_string);

    void enter();
    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable())
        {
            policy_thread.join();
        }
        if (telemetry_log_.is_open())
        {
            telemetry_log_.close();
        }
    }

    // Shared with the observation terms (jump_command / jump_time) of the
    // currently active flip state. Set on enter(), like State_Mimic::motion.
    static std::shared_ptr<FlipCommand> command;

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<FlipCommand> command_;

    std::string trigger_key_;             // (single-shot) optional key to fire the one configured motion
    std::vector<MotionPreset> motions_;   // (manual/Dynamic) key -> motion presets
    bool manual_mode_ = false;            // true when a `motions:` list is configured
    // Grace period after the motion window before fall -> Passive. Must outlast the
    // whole landing *and* settle, not just the command: a backflip touches down around
    // command_duration_s + 0.3s (training's minimum_landing_time_s is 0.80s for a 0.50s
    // command), so a shorter delay un-gates the check at the exact instant the front
    // feet slam down and the robot is still pitching.
    float fall_check_delay_s_ = 1.2f;
    float bad_orientation_limit_ = 1.2f;
    // The tilt must exceed the limit continuously for this long before we call it a
    // fall. See the check in State_Flip.cpp for why a single sample is not enough.
    float fall_check_hold_s_ = 0.1f;
    std::chrono::steady_clock::time_point bad_orientation_since_{};
    bool bad_orientation_latched_ = false;

    std::thread policy_thread;
    bool policy_thread_running = false;

    // --- diagnostic telemetry (sim2sim gap investigation) -------------------
    // Logs one row per policy step while this state is active: pitch rotation
    // (integrated from root_ang_vel_b, mirroring the training-side
    // accumulated_pitch computation so it's directly comparable to IsaacLab's
    // Metrics/jump numbers), tilt, and per-joint velocity/torque, so a MuJoCo
    // run can be quantitatively compared against Isaac Sim's play-mode trace
    // instead of relying on visual observation alone.
    std::ofstream telemetry_log_;
    long telemetry_step_ = 0;
    float accumulated_pitch_deg_ = 0.0f;
    // Integrated at the 1 kHz capture rate and reset on each trigger, mirroring
    // JumpCommand.accumulated_roll / accumulated_pitch. Written to the torque CSV in turns.
    float capture_roll_rad_ = 0.0f;
    float capture_pitch_rad_ = 0.0f;
    void log_telemetry_row();

    // --- landing-impact diagnostics -----------------------------------------
    // Sampled from run(), i.e. at the 1 kHz FSM rate, not from the 50 Hz policy
    // loop: a touchdown spike lasts a few milliseconds, so policy-rate sampling
    // can miss its peak entirely -- and that peak is precisely the quantity in
    // question when a landing kicks the state machine out of this state. Peaks are
    // accumulated per motion and reported once, so a run that *doesn't* trip the
    // fall check still tells you how hard it landed (e.g. for comparing a backflip
    // against a sideflip).
    void sample_diagnostics();
    void report_impact() const;
    std::string impact_summary() const;
    float tilt_deg() const;

    // --- fusion-vs-gyro attitude check ---------------------------------------
    // tilt_deg() reads the IMU's fused orientation (gyro + accelerometer). This is a
    // second, independent estimate built by integrating ONLY the gyro (root_ang_vel_b),
    // seeded from the fused quaternion at the moment each capture starts, so any gap that
    // opens up between the two after that is the fusion drifting/lagging relative to the
    // gyro's own (bias-prone but not amplitude-limited) view of the rotation -- exactly the
    // failure mode suspected from the sim2real torque comparison: a fused attitude that
    // reports a calmer tilt than reality during the ~-19 rad/s sideflip spin, so the policy
    // stops correcting too early. Both columns are written to the torque CSV so this can be
    // checked directly against a real capture instead of only argued from indirect evidence
    // (the policy's own torque output).
    Eigen::Quaternionf gyro_dead_reckon_quat_ = Eigen::Quaternionf::Identity();
    static float tilt_from_quat(const Eigen::Quaternionf & quat);

    long diag_trigger_step_ = -1;      // which motion the peaks below belong to
    bool diag_reported_ = false;
    float peak_accel_ = 0.0f;          // |IMU acceleration| over the motion, m/s^2
    float peak_accel_time_s_ = 0.0f;
    float peak_tau_ = 0.0f;            // |tau_est| over the motion, Nm
    int peak_tau_motor_ = -1;          // SDK motor index of the above
    float peak_joint_vel_ = 0.0f;      // |dq| over the motion, rad/s
    // Tilt is only meaningful once the fall guard arms -- before that the robot is
    // deliberately upside down. Both are measured against bad_orientation_limit_.
    float tilt_at_arm_deg_ = -1.0f;    // single sample, taken as the guard arms
    float peak_tilt_deg_ = 0.0f;       // peak after the guard armed
    float peak_tilt_time_s_ = 0.0f;

    // Policy-loop health. The loop targets step_dt per iteration but does ONNX
    // inference and a flushed CSV write inline, either of which can block longer
    // than that on the robot's own filesystem/CPU -- and once an iteration overruns,
    // sleep_until stops sleeping and the effective control rate drops. That failure
    // cannot happen in simulation, so it is worth measuring separately from the
    // orientation guard. Written by the policy thread, read by the FSM thread.
    std::atomic<int> policy_overrun_count_{0};
    std::atomic<float> policy_step_max_ms_{0.0f};

    // --- high-rate torque capture -------------------------------------------
    // A time series, where the block above keeps only peaks: one row per FSM tick
    // (1 kHz) over a window around each motion, written to its own CSV. The 50 Hz
    // telemetry_log_ cannot do this job -- a push-off lasts ~0.15s, which is about
    // seven policy-rate samples, too few to show either the shape of the torque
    // curve or its peak.
    //
    // BOTH the commanded and the applied torque are recorded, and that is the point
    // of the whole capture. The applied value alone shows a joint pinned flat at its
    // limit but not by how far it is over: a controller asking for 46 N*m and one
    // asking for 200 N*m produce identical saturated traces, and the difference
    // between those two is exactly whether more torque would buy more height.
    // tau_cmd is the same expression the MuJoCo bridge evaluates
    // (unitree_sdk2_bridge.h:186) and the robot's motor firmware applies, so
    // tau_cmd - tau_app is the amount the clamp threw away.
    struct TorqueSample
    {
        long step;          // FSM tick; converted to seconds-from-trigger at write time
        float cmd_elapsed;  // command_->elapsed(), i.e. the 50 Hz policy clock
        int enabled;
        float base_z;       // world height of the IMU site; see base_height_ below
        // Attitude, without which a failed flip cannot be told apart from a successful
        // one: torque and joint angles look much the same whether the robot completed the
        // rotation, stopped short and landed on its flank, or over-rotated past vertical.
        // gravity_b is the same projected_gravity the policy observes -- z is -1 upright,
        // 0 on its side, +1 inverted -- and roll_turns integrates the roll rate exactly as
        // JumpCommand.accumulated_roll does, so both are directly comparable to Isaac Lab.
        float gravity_b[3];
        float ang_vel_b[3];
        float roll_turns;
        float pitch_turns;
        float tilt_deg;
        // Gyro-only dead-reckoned tilt (see gyro_dead_reckon_quat_): diverges from tilt_deg
        // above exactly when the fused estimate is lagging or amplitude-limited relative to
        // what the gyro alone says happened.
        float tilt_deg_gyro;
        float q[12];
        float dq[12];
        float tau_cmd[12];
        float tau_app[12];
    };

    void capture_torque_sample();
    void write_torque_capture();

    // True body height, for comparing the achieved jump against Isaac Lab's
    // Metrics/jump/max_height. Nothing in LowState carries position -- a real robot does
    // not know its own height -- but unitree_mujoco publishes MuJoCo's ground truth on
    // rt/sportmodestate: go2.xml declares a `frame_pos` sensor on the imu site and
    // unitree_sdk2_bridge.h copies it into SportModeState::position. So this column is
    // populated in simulation and stays 0 on hardware once the built-in sport service has
    // been released, which is the intended scope: it exists to check the sim2sim gap.
    //
    // The origin is the imu site rather than the base frame, so the absolute value carries
    // a fixed offset. Height is therefore reported relative to the pre-trigger baseline,
    // where that offset cancels.
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState> base_height_;

    bool torque_log_enabled_ = false;   // opt-in via `torque_log: true` in config.yaml
    float torque_pre_s_ = 0.3f;         // seconds kept before the trigger, as a baseline
    float torque_post_s_ = 1.5f;        // seconds kept after it -- must outlast the landing

    long fsm_step_ = 0;
    long torque_trigger_step_ = -1;     // command_->trigger_step this capture belongs to
    long torque_trigger_fsm_step_ = 0;
    bool torque_capturing_ = false;
    int torque_capture_index_ = 0;
    std::string torque_motion_;         // latched at trigger: targets are zeroed on re-arm
    std::vector<TorqueSample> torque_pre_;      // rolling pre-trigger baseline
    size_t torque_pre_head_ = 0;
    std::vector<TorqueSample> torque_capture_;
};

REGISTER_FSM(State_Flip)
