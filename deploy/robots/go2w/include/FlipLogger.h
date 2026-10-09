// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// FlipLogger: sim2sim / sim2real diagnostics for State_Flip, ported from the Go2 controller
// on feat/jump and widened to Go2W's 16 motors (12 legs + 4 wheels). Three outputs, the
// files in `log_dir` (config.yaml; default deploy/robots/go2w/output):
//
//   telemetry_<stamp>.csv       one row per policy step (50 Hz) while the state is active:
//                               command, attitude, integrated pitch/roll, IMU acceleration,
//                               per-motor dq / tau_est.
//   torque_<motion>_<stamp>_<n>.csv
//                               one row per FSM tick (1 kHz) in a window around each motion
//                               (opt-in, `torque_log: true`): q, dq, commanded AND applied
//                               torque per motor, attitude, body height. A push-off lasts
//                               ~0.15 s, about seven policy-rate samples -- too few for the
//                               shape or the peak of a torque curve.
//   log line per motion         peak |accel| / |tau| / |dq|, tilt once the fall guard arms,
//                               policy-loop overruns -- also printed when a fall trips the
//                               guard, so a run that drops to Passive says how hard it hit.
//
// Columns are in SDK motor order (FR, FL, RR, RL legs, then FR, FL, RR, RL wheels).

#pragma once

#include "Types.h"
#include "State_Flip.h"

#include <Eigen/Geometry>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

class FlipLogger
{
public:
    static constexpr int kNumMotors = 16;

    struct Config
    {
        // Relative paths resolve against the controller's project dir (deploy/robots/go2w),
        // like policy_dir. Created if missing.
        std::filesystem::path log_dir = "output";
        bool telemetry = true;
        bool torque_log = false;
        float torque_pre_s = 0.3f;   // kept before the trigger, as a standing baseline
        float torque_post_s = 1.5f;  // kept after it -- must outlast the landing
    };

    FlipLogger(Config cfg, std::shared_ptr<LowState_t> lowstate, LowCmd_t * lowcmd);

    // State lifecycle.
    void on_enter(const State_Flip::FlipCommand * command, float step_dt, float guard_arm_s);
    void on_exit();

    // Policy thread, once per policy step after env->step(). work_ms is the step's own cost.
    void on_policy_step(float work_ms);

    // FSM thread, once per run() tick (1 kHz).
    void on_fsm_tick();

    // One-line summary of the current motion's peaks (also used by the fall guard).
    std::string impact_summary() const;

private:
    struct TorqueSample
    {
        long step;          // FSM tick; converted to seconds-from-trigger at write time
        float cmd_elapsed;  // the 50 Hz policy clock
        int enabled;
        float base_z;       // world height of the IMU site (MuJoCo ground truth; 0 on hardware)
        // gravity_b is the projected_gravity the policy observes (z = -1 upright, +1
        // inverted); roll/pitch_turns integrate the body rates exactly as JumpCommand's
        // accumulated_roll/pitch do, so both compare directly against Isaac Lab.
        float gravity_b[3];
        float ang_vel_b[3];
        float roll_turns;
        float pitch_turns;
        float tilt_deg;
        // Gyro-only dead-reckoned tilt, seeded from the fused estimate at the trigger: a gap
        // from tilt_deg is the IMU fusion lagging the gyro's own view of a fast rotation.
        float tilt_deg_gyro;
        float q[kNumMotors];
        float dq[kNumMotors];
        // Commanded torque (the PD law the MuJoCo bridge and the motor firmware evaluate)
        // next to the applied one: a joint pinned at its limit looks the same whether the
        // controller asked for 46 or 200 N*m; tau_cmd - tau_app is what the clamp threw away.
        float tau_cmd[kNumMotors];
        float tau_app[kNumMotors];
    };

    float tilt_deg() const;
    static float tilt_from_quat(const Eigen::Quaternionf & quat);
    void sample_impact();
    void report_impact() const;
    void capture_torque_sample();
    void write_torque_capture();
    std::string motion_name() const;

    Config cfg_;
    std::shared_ptr<LowState_t> lowstate_;
    LowCmd_t * lowcmd_;
    const State_Flip::FlipCommand * command_ = nullptr;
    float step_dt_ = 0.02f;
    float guard_arm_s_ = 2.0f;  // command_duration_s + fall_check_delay_s

    // Ground-truth body height, published by unitree_mujoco on rt/sportmodestate (go2w.xml's
    // `frame_pos` sensor on the imu site). Stays 0 on hardware once the sport service is
    // released. Reported relative to the pre-trigger baseline, where the site offset cancels.
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState> base_height_;

    // --- telemetry (50 Hz) ---
    std::ofstream telemetry_;
    long telemetry_step_ = 0;
    float telemetry_pitch_deg_ = 0.0f;
    float telemetry_roll_deg_ = 0.0f;

    // --- impact peaks (1 kHz) ---
    long diag_trigger_step_ = -1;
    bool diag_reported_ = false;
    float peak_accel_ = 0.0f;
    float peak_accel_time_s_ = 0.0f;
    float peak_tau_ = 0.0f;
    int peak_tau_motor_ = -1;
    float peak_joint_vel_ = 0.0f;  // legs only: the wheels spin freely by design
    float tilt_at_arm_deg_ = -1.0f;
    float peak_tilt_deg_ = 0.0f;
    float peak_tilt_time_s_ = 0.0f;
    // Policy-loop health: ONNX inference and the flushed CSV write run inline and can
    // overrun step_dt on the robot's CPU/filesystem -- a failure simulation cannot show.
    std::atomic<int> policy_overrun_count_{0};
    std::atomic<float> policy_step_max_ms_{0.0f};

    // --- torque capture (1 kHz) ---
    long fsm_step_ = 0;
    long torque_trigger_step_ = -1;
    long torque_trigger_fsm_step_ = 0;
    bool torque_capturing_ = false;
    int torque_capture_index_ = 0;
    std::string torque_motion_;  // latched at the trigger: targets are zeroed on re-arm
    std::vector<TorqueSample> torque_pre_;
    size_t torque_pre_head_ = 0;
    std::vector<TorqueSample> torque_capture_;
    float capture_roll_rad_ = 0.0f;
    float capture_pitch_rad_ = 0.0f;
    Eigen::Quaternionf gyro_dead_reckon_quat_ = Eigen::Quaternionf::Identity();
};
