// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#include "BipedLogger.h"
#include "param.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <mutex>

namespace
{

constexpr float kFsmDt = 0.001f;  // CtrlFSM runs run() at 1 kHz
constexpr int kNumLegMotors = 12;
constexpr float kRadToDeg = 180.0f / static_cast<float>(M_PI);

const char * kSdkMotorNames[BipedLogger::kNumMotors] = {
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "FR_wheel", "FL_wheel", "RR_wheel", "RL_wheel",
};

std::string timestamp()
{
    char stamp[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    return stamp;
}

const char * motor_name(int i)
{
    return (i >= 0 && i < BipedLogger::kNumMotors) ? kSdkMotorNames[i] : "n/a";
}

} // namespace

BipedLogger::BipedLogger(Config cfg, std::string state_name, std::shared_ptr<LowState_t> lowstate, LowCmd_t * lowcmd)
: cfg_(std::move(cfg)), state_name_(std::move(state_name)), lowstate_(std::move(lowstate)), lowcmd_(lowcmd)
{
    if (cfg_.log_dir.is_relative())
    {
        cfg_.log_dir = param::proj_dir / cfg_.log_dir;
    }
    std::filesystem::create_directories(cfg_.log_dir);
    if (cfg_.telemetry || cfg_.torque_log)
    {
        base_height_ = std::make_shared<unitree::robot::go2::subscription::SportModeState>();
    }
    spdlog::info("BipedLogger[{}]: 50Hz telemetry {}, 1kHz torque capture {} ({:.1f}s from entry) -> {}",
                 state_name_, cfg_.telemetry ? "ON" : "off", cfg_.torque_log ? "ON" : "off",
                 cfg_.torque_log_s, cfg_.log_dir.string());
}

BipedLogger::~BipedLogger()
{
    if (torque_writer_.joinable())
    {
        torque_writer_.join();
    }
}

float BipedLogger::tilt_deg() const
{
    // The angle bad_orientation() thresholds, read from lowstate at the 1 kHz FSM rate.
    const auto & quat = lowstate_->msg_.imu_state().quaternion();
    const Eigen::Quaternionf root_quat(quat[0], quat[1], quat[2], quat[3]);
    const float grav_z = (root_quat.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f)).z();
    return std::acos(std::clamp(-grav_z, -1.0f, 1.0f)) * kRadToDeg;
}

float BipedLogger::base_z() const
{
    return base_height_ ? static_cast<float>(base_height_->msg_.position()[2]) : 0.0f;
}

void BipedLogger::on_enter(float stance, float step_dt)
{
    stance_ = stance;
    step_dt_ = step_dt;
    stamp_ = timestamp();
    telemetry_step_ = 0;
    accumulated_pitch_deg_ = 0.0f;
    fsm_step_ = 0;
    torque_written_ = false;
    torque_capture_.clear();
    if (cfg_.torque_log)
    {
        // Re-reserved every entry (the previous buffer was moved to the writer thread), so the
        // 1 kHz push_back never reallocates inside an FSM tick.
        torque_capture_.reserve(static_cast<size_t>(cfg_.torque_log_s / kFsmDt) + 1);
    }
    peak_tilt_deg_ = peak_tilt_time_s_ = 0.0f;
    peak_tau_ = peak_clamp_ = peak_track_err_ = peak_joint_vel_ = 0.0f;
    peak_tau_motor_ = peak_clamp_motor_ = peak_track_err_motor_ = -1;
    policy_overrun_count_.store(0);
    policy_step_max_ms_.store(0.0f);
    policy_step_last_ms_.store(0.0f);

    if (!cfg_.telemetry)
    {
        return;
    }
    const auto path = cfg_.log_dir / (state_name_ + "_telemetry_" + stamp_ + ".csv");
    telemetry_.open(path, std::ios::out | std::ios::trunc);
    if (!telemetry_)
    {
        spdlog::warn("BipedLogger[{}]: could not open {}", state_name_, path.string());
        return;
    }
    // step_ms is the previous step's loop time (stamped after the row is written).
    telemetry_ << "t,stance,tilt_deg,accumulated_pitch_deg,grav_x,grav_y,grav_z,gyro_x,gyro_y,gyro_z,"
               << "cmd_vx,cmd_vy,cmd_wz,base_z,step_ms";
    for (const char * field : {"q", "dq", "cmd", "tau_est"})
    {
        for (const char * name : kSdkMotorNames) telemetry_ << "," << name << "_" << field;
    }
    telemetry_ << "\n";
    spdlog::info("BipedLogger[{}]: telemetry -> {}", state_name_, path.string());
}

void BipedLogger::on_exit()
{
    if (telemetry_.is_open())
    {
        telemetry_.close();
    }
    write_torque_capture();  // no-op if the window already closed and wrote itself out
    if (torque_writer_.joinable())
    {
        torque_writer_.join();
    }
    report_summary();
}

void BipedLogger::on_policy_step(const std::vector<float> & command, float work_ms)
{
    if (work_ms > policy_step_max_ms_.load()) policy_step_max_ms_.store(work_ms);
    if (work_ms > step_dt_ * 1e3f) policy_overrun_count_.fetch_add(1);

    if (!telemetry_.is_open())
    {
        policy_step_last_ms_.store(work_ms);
        return;
    }

    // Copied under the lowstate lock (the FSM thread updates it at 1 kHz), formatted outside.
    float gyro[3], quat[4], q[kNumMotors], dq[kNumMotors], cmd[kNumMotors], tau[kNumMotors];
    {
        std::lock_guard<std::mutex> lock(lowstate_->mutex_);
        const auto & imu = lowstate_->msg_.imu_state();
        for (int i = 0; i < 3; ++i) gyro[i] = imu.gyroscope()[i];
        for (int i = 0; i < 4; ++i) quat[i] = imu.quaternion()[i];
        for (int i = 0; i < kNumMotors; ++i)
        {
            const auto & motor = lowstate_->msg_.motor_state()[i];
            const auto & motor_cmd = lowcmd_->msg_.motor_cmd()[i];
            q[i] = motor.q();
            dq[i] = motor.dq();
            cmd[i] = i < kNumLegMotors ? motor_cmd.q() : motor_cmd.dq();
            tau[i] = motor.tau_est();
        }
    }
    const Eigen::Quaternionf root_quat(quat[0], quat[1], quat[2], quat[3]);
    const Eigen::Vector3f gravity = root_quat.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    const float tilt = std::acos(std::clamp(-gravity.z(), -1.0f, 1.0f)) * kRadToDeg;
    // Signed, unlike tilt: a front stance (nose down) and a hind stance (nose up) integrate to
    // opposite signs, so a rise that starts the wrong way shows up at once.
    accumulated_pitch_deg_ += gyro[1] * step_dt_ * kRadToDeg;

    telemetry_ << (telemetry_step_ * step_dt_) << "," << stance_ << "," << tilt << ","
               << accumulated_pitch_deg_ << "," << gravity.x() << "," << gravity.y() << ","
               << gravity.z() << "," << gyro[0] << "," << gyro[1] << "," << gyro[2];
    for (int i = 0; i < 3; ++i)
    {
        telemetry_ << "," << (i < static_cast<int>(command.size()) ? command[i] : 0.0f);
    }
    telemetry_ << "," << base_z() << "," << policy_step_last_ms_.load();
    for (int i = 0; i < kNumMotors; ++i) telemetry_ << "," << q[i];
    for (int i = 0; i < kNumMotors; ++i) telemetry_ << "," << dq[i];
    for (int i = 0; i < kNumMotors; ++i) telemetry_ << "," << cmd[i];
    for (int i = 0; i < kNumMotors; ++i) telemetry_ << "," << tau[i];
    telemetry_ << "\n";
    // Flushed per row: an attempt that ends in a fall is often killed by hand, and a buffered
    // tail would lose exactly the part being investigated. The cost shows in step_ms.
    telemetry_.flush();
    telemetry_step_ += 1;
    policy_step_last_ms_.store(work_ms);
}

void BipedLogger::on_fsm_tick()
{
    fsm_step_ += 1;
    const float t = fsm_step_ * kFsmDt;

    const float tilt = tilt_deg();
    if (tilt > peak_tilt_deg_)
    {
        peak_tilt_deg_ = tilt;
        peak_tilt_time_s_ = t;
    }

    const bool capture = cfg_.torque_log && !torque_written_ && t <= cfg_.torque_log_s;
    TorqueSample s;
    if (capture)
    {
        s.step = fsm_step_;
        s.tilt_deg = tilt;
        s.base_z = base_z();
    }

    for (int i = 0; i < kNumMotors; ++i)
    {
        const auto & motor = lowstate_->msg_.motor_state()[i];
        const auto & motor_cmd = lowcmd_->msg_.motor_cmd()[i];
        // The PD law the motor firmware and the MuJoCo bridge (unitree_sdk2_bridge.h) apply.
        const float tau_cmd = motor_cmd.tau() + motor_cmd.kp() * (motor_cmd.q() - motor.q())
                              + motor_cmd.kd() * (motor_cmd.dq() - motor.dq());
        const float tau_app = motor.tau_est();

        if (std::fabs(tau_app) > peak_tau_)
        {
            peak_tau_ = std::fabs(tau_app);
            peak_tau_motor_ = i;
        }
        const float clamped = std::fabs(tau_cmd) - std::fabs(tau_app);
        if (clamped > peak_clamp_)
        {
            peak_clamp_ = clamped;
            peak_clamp_motor_ = i;
        }
        if (i < kNumLegMotors)
        {
            const float track_err = std::fabs(motor_cmd.q() - motor.q());
            if (track_err > peak_track_err_)
            {
                peak_track_err_ = track_err;
                peak_track_err_motor_ = i;
            }
            peak_joint_vel_ = std::max(peak_joint_vel_, std::fabs(motor.dq()));
        }

        if (capture)
        {
            s.q[i] = motor.q();
            s.dq[i] = motor.dq();
            s.cmd[i] = i < kNumLegMotors ? motor_cmd.q() : motor_cmd.dq();
            s.tau_cmd[i] = tau_cmd;
            s.tau_app[i] = tau_app;
        }
    }

    if (capture)
    {
        torque_capture_.push_back(s);
        if (t >= cfg_.torque_log_s)
        {
            // Written as soon as the window closes, so it is on disk even if the run is killed.
            write_torque_capture();
        }
    }
}

void BipedLogger::write_torque_capture()
{
    if (torque_written_ || torque_capture_.empty())
    {
        return;
    }
    torque_written_ = true;
    const auto path = cfg_.log_dir / (state_name_ + "_torque_" + stamp_ + ".csv");
    if (torque_writer_.joinable())
    {
        torque_writer_.join();
    }
    torque_writer_ = std::thread(&BipedLogger::write_torque_csv, path.string(), std::move(torque_capture_));
    torque_capture_.clear();  // moved-from; back to a defined empty state
}

void BipedLogger::write_torque_csv(std::string path, std::vector<TorqueSample> samples)
{
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out)
    {
        spdlog::warn("BipedLogger: could not open {}", path);
        return;
    }
    out << "t,tilt_deg,base_z";
    for (const char * field : {"q", "dq", "cmd", "tau_cmd", "tau_app"})
    {
        for (const char * name : kSdkMotorNames) out << "," << name << "_" << field;
    }
    out << "\n";
    for (const auto & s : samples)
    {
        out << s.step * kFsmDt << "," << s.tilt_deg << "," << s.base_z;
        for (float v : s.q) out << "," << v;
        for (float v : s.dq) out << "," << v;
        for (float v : s.cmd) out << "," << v;
        for (float v : s.tau_cmd) out << "," << v;
        for (float v : s.tau_app) out << "," << v;
        out << "\n";
    }
    spdlog::info("BipedLogger: wrote {} ({} rows, 0.00 .. {:.2f}s from entry)",
                 path, samples.size(), samples.empty() ? 0.0f : samples.back().step * kFsmDt);
}

void BipedLogger::report_summary() const
{
    spdlog::info(
        "BipedLogger[{}]: {:.1f}s active, peak tilt {:.1f} deg @{:.2f}s (the stance is ~70-90 deg) -- "
        "peak |tau| {:.1f} Nm ({}), peak clamped torque {:.1f} Nm ({}), "
        "peak leg |q_des - q| {:.3f} rad ({}), peak leg |dq| {:.1f} rad/s -- "
        "policy step max {:.1f} ms of {:.1f} ms budget, {} overrun(s)",
        state_name_, fsm_step_ * kFsmDt, peak_tilt_deg_, peak_tilt_time_s_,
        peak_tau_, motor_name(peak_tau_motor_), peak_clamp_, motor_name(peak_clamp_motor_),
        peak_track_err_, motor_name(peak_track_err_motor_), peak_joint_vel_,
        policy_step_max_ms_.load(), step_dt_ * 1e3f, policy_overrun_count_.load());
}
