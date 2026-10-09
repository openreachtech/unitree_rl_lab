// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#include "FlipLogger.h"
#include "param.h"

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <ctime>

namespace
{

constexpr float kFsmDt = 0.001f;  // CtrlFSM runs run() at 1 kHz
constexpr int kNumLegMotors = 12;

const char * kSdkMotorNames[FlipLogger::kNumMotors] = {
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

} // namespace

FlipLogger::FlipLogger(Config cfg, std::shared_ptr<LowState_t> lowstate, LowCmd_t * lowcmd)
: cfg_(cfg), lowstate_(std::move(lowstate)), lowcmd_(lowcmd)
{
    if (cfg_.log_dir.is_relative())
    {
        cfg_.log_dir = param::proj_dir / cfg_.log_dir;
    }
    std::filesystem::create_directories(cfg_.log_dir);
    spdlog::info("FlipLogger: writing logs to {}", cfg_.log_dir.string());
    if (cfg_.torque_log)
    {
        // Constructed regardless of whether anything publishes, so the column always exists.
        base_height_ = std::make_shared<unitree::robot::go2::subscription::SportModeState>();
        spdlog::info("FlipLogger: 1kHz torque capture ON -- torque_<motion>_*.csv, window [-{:.2f}s, +{:.2f}s]",
                     cfg_.torque_pre_s, cfg_.torque_post_s);
    }
}

void FlipLogger::on_enter(const State_Flip::FlipCommand * command, float step_dt, float guard_arm_s)
{
    command_ = command;
    step_dt_ = step_dt;
    guard_arm_s_ = guard_arm_s;
    diag_trigger_step_ = -1;
    diag_reported_ = false;
    torque_trigger_step_ = -1;
    torque_capturing_ = false;
    torque_pre_.clear();
    torque_pre_head_ = 0;

    if (!cfg_.telemetry)
    {
        return;
    }
    telemetry_step_ = 0;
    telemetry_pitch_deg_ = 0.0f;
    telemetry_roll_deg_ = 0.0f;
    const std::string path = (cfg_.log_dir / ("telemetry_" + timestamp() + ".csv")).string();
    telemetry_.open(path, std::ios::out | std::ios::trunc);
    telemetry_ << "t,enabled,elapsed_since_trigger,target_height,target_pitch_turns,target_roll_turns,"
               << "accumulated_pitch_deg,accumulated_roll_deg,grav_x,grav_y,grav_z,tilt_deg,"
               << "accel_x,accel_y,accel_z,wheel_brake";
    for (const char * name : kSdkMotorNames) telemetry_ << "," << name << "_dq";
    for (const char * name : kSdkMotorNames) telemetry_ << "," << name << "_tau";
    telemetry_ << "\n";
    spdlog::info("FlipLogger: telemetry -> {}", path);
}

void FlipLogger::on_exit()
{
    if (torque_capturing_)
    {
        write_torque_capture();
    }
    if (diag_trigger_step_ >= 0 && !diag_reported_)
    {
        report_impact();
        diag_reported_ = true;
    }
    if (telemetry_.is_open())
    {
        telemetry_.close();
    }
}

void FlipLogger::on_policy_step(float work_ms, bool wheel_brake)
{
    if (work_ms > policy_step_max_ms_.load()) policy_step_max_ms_.store(work_ms);
    if (work_ms > step_dt_ * 1e3f) policy_overrun_count_.fetch_add(1);

    if (!telemetry_.is_open() || !command_)
    {
        return;
    }
    const auto & imu = lowstate_->msg_.imu_state();
    // Integrated like JumpCommand.accumulated_pitch/roll so it compares to Isaac Lab directly.
    const float rad2deg = 180.0f / static_cast<float>(M_PI);
    telemetry_roll_deg_ += imu.gyroscope()[0] * step_dt_ * rad2deg;
    telemetry_pitch_deg_ += imu.gyroscope()[1] * step_dt_ * rad2deg;

    const Eigen::Quaternionf quat(imu.quaternion()[0], imu.quaternion()[1], imu.quaternion()[2], imu.quaternion()[3]);
    const Eigen::Vector3f gravity = quat.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);

    telemetry_ << (telemetry_step_ * step_dt_) << "," << (command_->enabled ? 1 : 0) << ","
               << command_->elapsed() << "," << command_->target_height << ","
               << command_->target_pitch_turns << "," << command_->target_roll_turns << ","
               << telemetry_pitch_deg_ << "," << telemetry_roll_deg_ << ","
               << gravity.x() << "," << gravity.y() << "," << gravity.z() << "," << tilt_deg();
    for (int i = 0; i < 3; ++i) telemetry_ << "," << imu.accelerometer()[i];
    telemetry_ << "," << (wheel_brake ? 1 : 0);
    for (int i = 0; i < kNumMotors; ++i) telemetry_ << "," << lowstate_->msg_.motor_state()[i].dq();
    for (int i = 0; i < kNumMotors; ++i) telemetry_ << "," << lowstate_->msg_.motor_state()[i].tau_est();
    telemetry_ << "\n";
    telemetry_.flush();
    telemetry_step_ += 1;
}

void FlipLogger::on_fsm_tick()
{
    if (!command_)
    {
        return;
    }
    sample_impact();
    capture_torque_sample();
}

float FlipLogger::tilt_from_quat(const Eigen::Quaternionf & quat)
{
    const float grav_z = (quat.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f)).z();
    return std::acos(std::clamp(-grav_z, -1.0f, 1.0f)) * 180.0f / static_cast<float>(M_PI);
}

float FlipLogger::tilt_deg() const
{
    // The angle bad_orientation() thresholds, read straight from lowstate so it is current at
    // the 1 kHz FSM rate rather than the policy rate.
    const auto & q = lowstate_->msg_.imu_state().quaternion();
    return tilt_from_quat(Eigen::Quaternionf(q[0], q[1], q[2], q[3]));
}

std::string FlipLogger::motion_name() const
{
    // Rotation before height: flips carry a height too (backflip 0.30, sideflip 0.50).
    if (command_->target_roll_turns != 0.0f) return "sideflip";
    if (command_->target_pitch_turns != 0.0f) return "backflip";
    if (command_->target_height != 0.0f) return "jump";
    return "motion";
}

std::string FlipLogger::impact_summary() const
{
    return fmt::format(
        "peak |accel| {:.1f} m/s^2 @{:.2f}s, peak |tau| {:.1f} Nm ({}), peak leg |dq| {:.1f} rad/s, "
        "tilt at guard arm {:.1f} deg, peak tilt after arm {:.1f} deg @{:.2f}s, "
        "policy step max {:.1f} ms of {:.1f} ms budget, {} overrun(s)",
        peak_accel_, peak_accel_time_s_, peak_tau_,
        peak_tau_motor_ >= 0 ? kSdkMotorNames[peak_tau_motor_] : "-",
        peak_joint_vel_, tilt_at_arm_deg_, peak_tilt_deg_, peak_tilt_time_s_,
        policy_step_max_ms_.load(), step_dt_ * 1e3f, policy_overrun_count_.load());
}

void FlipLogger::report_impact() const
{
    spdlog::info("FlipLogger: motion landed -- {}", impact_summary());
}

void FlipLogger::sample_impact()
{
    if (command_->trigger_step < 0)
    {
        return;
    }
    // A new motion: flush the previous one first, in case motions were chained faster than
    // the report delay.
    if (command_->trigger_step != diag_trigger_step_)
    {
        if (diag_trigger_step_ >= 0 && !diag_reported_)
        {
            report_impact();
        }
        diag_trigger_step_ = command_->trigger_step;
        diag_reported_ = false;
        peak_accel_ = peak_accel_time_s_ = peak_tau_ = peak_joint_vel_ = 0.0f;
        peak_tau_motor_ = -1;
        tilt_at_arm_deg_ = -1.0f;
        peak_tilt_deg_ = peak_tilt_time_s_ = 0.0f;
        policy_overrun_count_.store(0);
        policy_step_max_ms_.store(0.0f);
    }

    const float since_trigger = command_->elapsed();
    const auto & accel = lowstate_->msg_.imu_state().accelerometer();
    const float accel_norm = std::sqrt(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]);
    if (accel_norm > peak_accel_)
    {
        peak_accel_ = accel_norm;
        peak_accel_time_s_ = since_trigger;
    }
    for (int i = 0; i < kNumMotors; ++i)
    {
        const auto & motor = lowstate_->msg_.motor_state()[i];
        const float tau = std::fabs(motor.tau_est());
        if (tau > peak_tau_)
        {
            peak_tau_ = tau;
            peak_tau_motor_ = i;
        }
        if (i < kNumLegMotors)
        {
            peak_joint_vel_ = std::max(peak_joint_vel_, std::fabs(motor.dq()));
        }
    }

    // Tilt only means something once the fall guard arms -- before that the robot is
    // deliberately upside down.
    if (since_trigger >= guard_arm_s_)
    {
        const float tilt = tilt_deg();
        if (tilt_at_arm_deg_ < 0.0f) tilt_at_arm_deg_ = tilt;
        if (tilt > peak_tilt_deg_)
        {
            peak_tilt_deg_ = tilt;
            peak_tilt_time_s_ = since_trigger;
        }
    }
    if (!diag_reported_ && since_trigger >= guard_arm_s_ + 0.5f)
    {
        report_impact();
        diag_reported_ = true;
    }
}

void FlipLogger::capture_torque_sample()
{
    if (!cfg_.torque_log)
    {
        return;
    }
    fsm_step_ += 1;

    // A new motion opens a capture, seeded with the rolling pre-trigger buffer so the file
    // carries a standing baseline to measure the push-off against.
    const long ts = command_->trigger_step;
    if (ts >= 0 && ts != torque_trigger_step_)
    {
        if (torque_capturing_)
        {
            write_torque_capture();  // motions chained faster than torque_post_s
        }
        torque_trigger_step_ = ts;
        torque_trigger_fsm_step_ = fsm_step_;
        torque_motion_ = motion_name();
        capture_roll_rad_ = 0.0f;
        capture_pitch_rad_ = 0.0f;
        // Seeded from the fused estimate while the robot still stands and the two agree.
        const auto & q = lowstate_->msg_.imu_state().quaternion();
        gyro_dead_reckon_quat_ = Eigen::Quaternionf(q[0], q[1], q[2], q[3]);
        torque_capture_.clear();
        const size_t n = torque_pre_.size();
        for (size_t k = 0; k < n; ++k)
        {
            torque_capture_.push_back(torque_pre_[(torque_pre_head_ + k) % n]);
        }
        torque_capturing_ = true;
    }

    const auto & imu = lowstate_->msg_.imu_state();
    const Eigen::Quaternionf quat(imu.quaternion()[0], imu.quaternion()[1], imu.quaternion()[2], imu.quaternion()[3]);
    const Eigen::Vector3f gravity = quat.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    const Eigen::Vector3f omega(imu.gyroscope()[0], imu.gyroscope()[1], imu.gyroscope()[2]);

    TorqueSample s;
    s.step = fsm_step_;
    s.cmd_elapsed = command_->elapsed();
    s.enabled = command_->enabled ? 1 : 0;
    s.base_z = base_height_ ? static_cast<float>(base_height_->msg_.position()[2]) : 0.0f;
    for (int i = 0; i < 3; ++i)
    {
        s.gravity_b[i] = gravity[i];
        s.ang_vel_b[i] = omega[i];
    }
    if (torque_capturing_)
    {
        capture_roll_rad_ += omega.x() * kFsmDt;
        capture_pitch_rad_ += omega.y() * kFsmDt;
        // Full 3-axis body-rate integration (small-rotation increment, right-multiplied), so
        // it stays a valid attitude through the yaw/pitch coupling a real flip picks up.
        const Eigen::Vector3f omega_dt = omega * kFsmDt;
        const float angle = omega_dt.norm();
        if (angle > 1e-8f)
        {
            const Eigen::Quaternionf dq(Eigen::AngleAxisf(angle, omega_dt / angle));
            gyro_dead_reckon_quat_ = (gyro_dead_reckon_quat_ * dq).normalized();
        }
    }
    s.roll_turns = capture_roll_rad_ / (2.0f * static_cast<float>(M_PI));
    s.pitch_turns = capture_pitch_rad_ / (2.0f * static_cast<float>(M_PI));
    s.tilt_deg = tilt_from_quat(quat);
    s.tilt_deg_gyro = tilt_from_quat(gyro_dead_reckon_quat_);
    for (int i = 0; i < kNumMotors; ++i)
    {
        const auto & motor = lowstate_->msg_.motor_state()[i];
        const auto & cmd = lowcmd_->msg_.motor_cmd()[i];
        s.q[i] = motor.q();
        s.dq[i] = motor.dq();
        s.tau_app[i] = motor.tau_est();
        s.tau_cmd[i] = cmd.tau() + cmd.kp() * (cmd.q() - motor.q()) + cmd.kd() * (cmd.dq() - motor.dq());
    }

    if (torque_capturing_)
    {
        torque_capture_.push_back(s);
        if ((fsm_step_ - torque_trigger_fsm_step_) * kFsmDt >= cfg_.torque_post_s)
        {
            write_torque_capture();
        }
        return;
    }

    // Idle: keep a rolling window so the next trigger has a baseline to prepend.
    const size_t cap = static_cast<size_t>(cfg_.torque_pre_s / kFsmDt);
    if (cap == 0)
    {
        return;
    }
    if (torque_pre_.size() < cap)
    {
        torque_pre_.push_back(s);
    }
    else
    {
        torque_pre_[torque_pre_head_] = s;
        torque_pre_head_ = (torque_pre_head_ + 1) % cap;
    }
}

void FlipLogger::write_torque_capture()
{
    torque_capturing_ = false;
    // The next motion should be prefixed by the stand right before it, not by this landing.
    torque_pre_.clear();
    torque_pre_head_ = 0;
    if (torque_capture_.empty())
    {
        return;
    }

    // Height relative to the pre-trigger mean, which cancels the imu-site offset.
    float baseline_z = torque_capture_.front().base_z;
    {
        double sum = 0.0;
        int n = 0;
        for (const auto & s : torque_capture_)
        {
            if (s.step < torque_trigger_fsm_step_) { sum += s.base_z; ++n; }
        }
        if (n > 0) baseline_z = static_cast<float>(sum / n);
    }

    const std::string path = (cfg_.log_dir / ("torque_" + torque_motion_ + "_" + timestamp() + "_" +
                                              std::to_string(torque_capture_index_++) + ".csv")).string();
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out)
    {
        spdlog::warn("FlipLogger: could not open {} for torque capture", path);
        torque_capture_.clear();
        return;
    }

    out << "t,cmd_elapsed,enabled,base_z,height_delta"
           ",grav_x,grav_y,grav_z,wx,wy,wz,roll_turns,pitch_turns,tilt_deg,tilt_deg_gyro";
    for (const char * field : {"q", "dq", "tau_cmd", "tau_app"})
    {
        for (const char * name : kSdkMotorNames) out << "," << name << "_" << field;
    }
    out << "\n";

    float peak_delta = 0.0f;
    for (const auto & s : torque_capture_)
    {
        peak_delta = std::max(peak_delta, s.base_z - baseline_z);
        out << (s.step - torque_trigger_fsm_step_) * kFsmDt << "," << s.cmd_elapsed << "," << s.enabled
            << "," << s.base_z << "," << (s.base_z - baseline_z);
        for (int i = 0; i < 3; ++i) out << "," << s.gravity_b[i];
        for (int i = 0; i < 3; ++i) out << "," << s.ang_vel_b[i];
        out << "," << s.roll_turns << "," << s.pitch_turns << "," << s.tilt_deg << "," << s.tilt_deg_gyro;
        for (int i = 0; i < kNumMotors; ++i) out << "," << s.q[i];
        for (int i = 0; i < kNumMotors; ++i) out << "," << s.dq[i];
        for (int i = 0; i < kNumMotors; ++i) out << "," << s.tau_cmd[i];
        for (int i = 0; i < kNumMotors; ++i) out << "," << s.tau_app[i];
        out << "\n";
    }
    out.close();

    spdlog::info("FlipLogger: peak height above standing = {:.3f} m (compare Isaac Lab's "
                 "Metrics/jump/max_height; 0.000 means nothing published body position)", peak_delta);
    spdlog::info("FlipLogger: wrote {} ({} rows, {:.2f}s .. {:.2f}s around trigger)", path, torque_capture_.size(),
                 (torque_capture_.front().step - torque_trigger_fsm_step_) * kFsmDt,
                 (torque_capture_.back().step - torque_trigger_fsm_step_) * kFsmDt);
    torque_capture_.clear();
}
