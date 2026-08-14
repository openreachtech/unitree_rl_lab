#pragma once

#include "FSM/State_RLBase.h"
#include "GaitMode.h"

#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

/**
 * @brief RL state for a Go2 gait_mode-aware policy (e.g. Isaac Lab task `Go2-Biped-Phase1`).
 *
 * The network takes a discrete `gait_mode` one-hot observation alongside the usual
 * proprioception. Training currently pins that input to a single mode for the whole
 * episode (see `PinnedGaitModeCommand`) rather than resampling it, so the policy has
 * only ever seen `default_gait_mode` (hind_biped unless overridden in config.yaml).
 * The operator can still switch modes live -- through the joystick / keyboard bindings
 * under `FSM.<state>.gait_modes` and `FSM.<state>.keyboard_gait_modes` -- for exploring
 * how a pinned-mode checkpoint responds to a mode it was never trained to expect, but
 * only `default_gait_mode` is validated behavior.
 */
class State_BipedRL : public State_RLBase
{
public:
    State_BipedRL(int state_mode, std::string state_string);
    // Not an override -- BaseState has no virtual destructor. Reached anyway, because
    // CtrlFSM holds these as shared_ptr built by make_shared<Derived> (REGISTER_FSM),
    // which carries a deleter for the concrete type.
    ~State_BipedRL();

    void enter() override;
    void run() override;
    void exit() override;

protected:
    // The biped stances hold the base pitched ~70-90 degrees off flat, which the
    // quadruped tilt limit reads as a fall, so that limit only applies once the robot
    // is actually standing on four legs again -- see update_tilt_limit_arming().
    bool fall_detected() const override;

    void on_policy_step() override;

private:
    struct ModeTrigger
    {
        std::function<bool()> triggered;
        go2::GaitMode mode;
    };

    void load_mode_triggers(YAML::Node cfg, const std::string& state_string);
    void update_tilt_limit_arming();

    std::vector<ModeTrigger> mode_triggers_;

    // Mode commanded to the policy on entry, before any operator trigger fires.
    // Config key: FSM.<state>.default_gait_mode (name string, e.g. "hind_biped").
    go2::GaitMode default_gait_mode_ = go2::GaitMode::kHindBiped;

    // Only meant to catch a full flip; see fall_detected().
    float biped_tilt_limit_ = 2.6;

    // Whether the strict quadruped tilt limit is in force. Commanding quad only starts
    // the robot lowering itself; the limit is armed when it has actually settled.
    bool quad_tilt_limit_armed_ = false;

    // --- diagnostics: why a stance that works in Isaac Lab / MuJoCo does not rise on
    // --- hardware ------------------------------------------------------------------
    // The whole deploy stack (ONNX, deploy.yaml, gains, observation wiring) is shared
    // with the MuJoCo sim2sim run, so a stance that transfers to MuJoCo but not to the
    // robot differs only in physics. Deciding *which* physics needs the numbers below;
    // visual observation cannot separate "the policy never commands the rise" from
    // "it commands it and the joints do not follow".
    //
    // The summary (report_summary) is always produced -- it costs a handful of float
    // comparisons per FSM tick and is what makes a failed attempt quantitative. The two
    // CSVs are opt-in per state, since they write into the process CWD.
    std::string state_name_;
    std::string log_stamp_;   // shared by both files of one entry, so they pair up
    std::string log_dir_;     // config: log_dir (default: process CWD)

    // 50 Hz, one row per policy step for the whole time the state is active: what the
    // policy saw, what it asked for, and what the joints did. Config: telemetry.
    bool telemetry_enabled_ = false;
    std::ofstream telemetry_log_;
    long telemetry_step_ = 0;
    float accumulated_pitch_deg_ = 0.0f;
    void open_telemetry();
    void log_telemetry_row();

    // 1 kHz, a window starting at state entry, covering the rise attempt. The 50 Hz
    // trace cannot resolve a torque clamp: saturation lasting a few tens of ms is one
    // or two policy-rate samples. BOTH the commanded and the applied torque are kept,
    // and that pairing is the point -- tau_app alone shows a joint pinned at its limit
    // but not by how far it is over, and tau_cmd - tau_app is exactly what the motor
    // firmware's clamp threw away. Config: torque_log / torque_log_s.
    struct TorqueSample
    {
        long step;        // FSM tick, converted to seconds-from-entry at write time
        float tilt_deg;
        float base_z;     // published by unitree_mujoco only; 0 on hardware
        float q[12];
        float dq[12];
        float q_des[12];
        float tau_cmd[12];
        float tau_app[12];
    };

    bool torque_log_enabled_ = false;
    float torque_log_s_ = 8.0f;   // window from entry; must outlast the rise attempt
    std::vector<TorqueSample> torque_capture_;
    bool torque_written_ = false;
    void write_torque_capture();

    // The capture is buffered in memory and handed to this thread to write out, because
    // the window closes while the robot is still balancing: serializing thousands of rows
    // inline would block run(), i.e. stop publishing lowcmd, for as long as the write
    // takes. Takes everything by value so it holds no reference to this state.
    static void write_torque_csv(
        std::string path, std::string header, std::vector<TorqueSample> samples);
    std::thread torque_writer_;

    // Ground-truth body height, published by unitree_mujoco (go2.xml declares a
    // frame_pos sensor on the imu site, which the bridge copies into
    // SportModeState::position) and by nothing on hardware once the built-in sport
    // service is released. That asymmetry is the intended scope: the column exists so
    // a MuJoCo trace and a hardware trace can be overlaid where it *is* available.
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState> base_height_;

    // Peaks over one entry, sampled at the 1 kHz FSM rate (a clamp or a tracking spike
    // can be shorter than a policy step). Reported once, on exit.
    void sample_diagnostics();
    void reset_diagnostics();
    void report_summary() const;
    float tilt_deg() const;

    long fsm_step_ = 0;
    float peak_tilt_deg_ = 0.0f;       // "did it get up at all" -- a biped stance is ~70-90 deg
    float peak_tilt_time_s_ = 0.0f;
    float peak_tau_ = 0.0f;            // |tau_est|, Nm
    int peak_tau_joint_ = -1;          // SDK motor index of the above
    float peak_clamp_ = 0.0f;          // max |tau_cmd| - |tau_app|: torque the clamp refused
    int peak_clamp_joint_ = -1;
    float peak_track_err_ = 0.0f;      // max |q_des - q|, rad
    int peak_track_err_joint_ = -1;
    float peak_joint_vel_ = 0.0f;      // |dq|, rad/s
};

REGISTER_FSM(State_BipedRL)
