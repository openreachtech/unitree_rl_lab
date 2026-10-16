// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// BipedLogger: sim2sim / sim2real diagnostics for State_Biped, ported from the Go2 handstand
// controller on feat/biped (State_BipedRL) and widened to Go2W's 16 motors. Three outputs, the
// files in `log_dir` (config.yaml; default deploy/robots/go2w/output), one set per entry:
//
//   <state>_telemetry_<stamp>.csv  one row per policy step (50 Hz) while the state is active:
//                                  stance, tilt, integrated pitch, gravity, gyro, velocity
//                                  command, body height, step time, and per motor q / dq /
//                                  command / tau_est.
//   <state>_torque_<stamp>.csv     one row per FSM tick (1 kHz) over the first torque_log_s
//                                  seconds -- the rise. Commanded AND applied torque per motor:
//                                  the 50 Hz trace cannot resolve a torque clamp lasting a few
//                                  tens of ms, and tau_cmd - tau_app is what the clamp threw away.
//   log line on exit               peak tilt (did it get up at all -- the stance is ~70-90 deg),
//                                  peak |tau|, peak clamped torque, peak leg tracking error,
//                                  peak leg |dq|, policy-loop overruns.
//
// Columns are in SDK motor order (FR, FL, RR, RL legs, then FR, FL, RR, RL wheels). A motor's
// `cmd` is its position target for a leg and its velocity target for a wheel.

#pragma once

#include "Types.h"

#include <Eigen/Geometry>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

class BipedLogger
{
public:
    static constexpr int kNumMotors = 16;

    struct Config
    {
        // Relative paths resolve against the controller's project dir (deploy/robots/go2w),
        // like policy_dir. Created if missing.
        std::filesystem::path log_dir = "output";
        bool telemetry = true;
        bool torque_log = true;
        float torque_log_s = 8.0f;  // window from entry; must outlast the rise
    };

    BipedLogger(Config cfg, std::string state_name, std::shared_ptr<LowState_t> lowstate, LowCmd_t * lowcmd);
    ~BipedLogger();

    // State lifecycle.
    void on_enter(float stance, float step_dt);
    void on_exit();

    // Policy thread, once per policy step. command is the velocity command the policy saw;
    // work_ms is the step's own cost.
    void on_policy_step(const std::vector<float> & command, float work_ms);

    // FSM thread, once per run() tick (1 kHz), after the commands are written.
    void on_fsm_tick();

private:
    struct TorqueSample
    {
        long step;        // FSM tick, converted to seconds-from-entry at write time
        float tilt_deg;
        float base_z;     // published by unitree_mujoco only; 0 on hardware
        float q[kNumMotors];
        float dq[kNumMotors];
        float cmd[kNumMotors];
        float tau_cmd[kNumMotors];
        float tau_app[kNumMotors];
    };

    float tilt_deg() const;
    float base_z() const;
    void write_torque_capture();
    static void write_torque_csv(std::string path, std::vector<TorqueSample> samples);
    void report_summary() const;

    Config cfg_;
    std::string state_name_;
    std::shared_ptr<LowState_t> lowstate_;
    LowCmd_t * lowcmd_;
    std::string stamp_;
    float stance_ = 0.0f;
    float step_dt_ = 0.02f;

    // Ground-truth body height, published by unitree_mujoco on rt/sportmodestate (go2w.xml's
    // `frame_pos` sensor on the imu site); 0 on hardware once the sport service is released.
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState> base_height_;

    // --- telemetry (50 Hz) ---
    std::ofstream telemetry_;
    long telemetry_step_ = 0;
    float accumulated_pitch_deg_ = 0.0f;  // signed: front stance pitches one way, hind the other

    // --- torque capture (1 kHz) ---
    std::vector<TorqueSample> torque_capture_;
    bool torque_written_ = false;
    // The window closes while the robot is still balancing: serializing thousands of rows on
    // the FSM thread would stop lowcmd being published for as long as the write takes.
    std::thread torque_writer_;

    // --- peaks over one entry (1 kHz) ---
    long fsm_step_ = 0;
    float peak_tilt_deg_ = 0.0f;
    float peak_tilt_time_s_ = 0.0f;
    float peak_tau_ = 0.0f;
    int peak_tau_motor_ = -1;
    float peak_clamp_ = 0.0f;
    int peak_clamp_motor_ = -1;
    float peak_track_err_ = 0.0f;  // legs only: the wheels are velocity-controlled
    int peak_track_err_motor_ = -1;
    float peak_joint_vel_ = 0.0f;  // legs only: the wheels roll by design
    std::atomic<int> policy_overrun_count_{0};
    std::atomic<float> policy_step_max_ms_{0.0f};
    std::atomic<float> policy_step_last_ms_{0.0f};
};
