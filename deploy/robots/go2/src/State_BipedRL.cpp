#include "State_BipedRL.h"

#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "KeyboardVelocityCommand.h"
#include "param.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <map>
#include <mutex>
#include <utility>
#include <spdlog/spdlog.h>

namespace
{

// Margin below tilt_limit_ the base has to reach before the strict quadruped limit is
// armed, so that arming right on the boundary cannot trip on sensor noise alone [rad].
constexpr float kQuadTiltArmMargin = 0.1f;

// CtrlFSM calls run() at 1 kHz (CtrlFSM::dt).
constexpr float kFsmDt = 0.001f;

constexpr float kRadToDeg = 180.0f / static_cast<float>(M_PI);

// SDK motor order for Go2, used to name the CSV columns. `joint_ids_map` maps a policy
// joint index to one of these, so the files are labelled by real joint rather than by
// the policy's own ordering.
const char* kSdkJointNames[12] = {
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
};

std::string timestamp_now()
{
    char stamp[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    return stamp;
}

} // namespace

namespace isaaclab
{

// Stance commanded to the multimode policy, one-hot [quad, hind_biped, front_biped].
// Training samples it from GaitModeCommand; deployment takes it from the operator.
REGISTER_OBSERVATION(gait_mode)
{
    (void)env;
    (void)params;
    return go2::GaitModeSelector::instance().one_hot();
}

// The biped policies feed their state estimator a stacked proprioception history,
// exported as a second set of `<term>_hist` terms next to the current-step ones. Term
// names have to be unique in deploy.yaml, so each history term needs its own
// registration; it returns the same quantity as its current-step twin, and the
// observation manager's per-term ring buffer (history_length) does the stacking.
REGISTER_OBSERVATION(base_ang_vel_hist)
{
    return mdp::base_ang_vel(env, params);
}

REGISTER_OBSERVATION(projected_gravity_hist)
{
    return mdp::projected_gravity(env, params);
}

REGISTER_OBSERVATION(velocity_commands_hist)
{
    return mdp::velocity_commands(env, params);
}

REGISTER_OBSERVATION(keyboard_velocity_commands_hist)
{
    (void)params;
    return go2::keyboard_velocity_command(env);
}

REGISTER_OBSERVATION(gait_mode_hist)
{
    return gait_mode(env, params);
}

REGISTER_OBSERVATION(joint_pos_rel_hist)
{
    return mdp::joint_pos_rel(env, params);
}

REGISTER_OBSERVATION(joint_vel_rel_hist)
{
    return mdp::joint_vel_rel(env, params);
}

REGISTER_OBSERVATION(last_action_hist)
{
    return mdp::last_action(env, params);
}

} // namespace isaaclab

State_BipedRL::State_BipedRL(int state_mode, std::string state_string)
: State_RLBase(state_mode, state_string)
, state_name_(state_string)
{
    auto cfg = param::config["FSM"][state_string];

    if (cfg["biped_tilt_limit"].IsDefined())
    {
        biped_tilt_limit_ = cfg["biped_tilt_limit"].as<float>();
    }

    // --- diagnostics (see the member declarations in State_BipedRL.h) ---
    if (cfg["telemetry"].IsDefined())   telemetry_enabled_ = cfg["telemetry"].as<bool>();
    if (cfg["torque_log"].IsDefined())  torque_log_enabled_ = cfg["torque_log"].as<bool>();
    if (cfg["torque_log_s"].IsDefined()) torque_log_s_ = cfg["torque_log_s"].as<float>();
    if (cfg["log_dir"].IsDefined())     log_dir_ = cfg["log_dir"].as<std::string>();

    if (telemetry_enabled_ || torque_log_enabled_)
    {
        // Constructed for either log: the 50 Hz row carries the same column, and it is
        // how a MuJoCo trace gets a body height to compare a hardware trace against.
        base_height_ = std::make_shared<unitree::robot::go2::subscription::SportModeState>();
        if (torque_log_enabled_)
        {
            torque_capture_.reserve(static_cast<size_t>(torque_log_s_ / kFsmDt) + 1);
        }
        spdlog::info(
            "FSM: State_{} diagnostics -- 50Hz telemetry {}, 1kHz torque capture {} ({:.1f}s "
            "from entry), output dir '{}'",
            state_string,
            telemetry_enabled_ ? "ON" : "off",
            torque_log_enabled_ ? "ON" : "off",
            torque_log_s_,
            log_dir_.empty() ? std::filesystem::current_path().string() : log_dir_);
    }

    if (cfg["default_gait_mode"].IsDefined())
    {
        go2::GaitMode mode;
        const auto name = cfg["default_gait_mode"].as<std::string>();
        if (go2::parse_gait_mode(name, mode))
        {
            default_gait_mode_ = mode;
        }
        else
        {
            spdlog::warn("FSM: State_{} unknown default_gait_mode '{}', keeping {}",
                state_string, name, go2::gait_mode_name(default_gait_mode_));
        }
    }

    load_mode_triggers(cfg, state_string);

    if (mode_triggers_.empty())
    {
        spdlog::warn(
            "FSM: State_{} has no gait_modes / keyboard_gait_modes bindings; "
            "the policy will stay in {} mode.",
            state_string,
            go2::gait_mode_name(default_gait_mode_));
    }
}

void State_BipedRL::load_mode_triggers(YAML::Node cfg, const std::string& state_string)
{
    auto add = [&](const std::string& mode_name,
                   const std::string& spec,
                   const char* source,
                   std::function<bool()> triggered)
    {
        go2::GaitMode mode;
        if (!go2::parse_gait_mode(mode_name, mode))
        {
            spdlog::warn("FSM: State_{} unknown gait mode '{}'", state_string, mode_name);
            return;
        }
        mode_triggers_.push_back({std::move(triggered), mode});
        spdlog::info(
            "FSM: State_{} {} gait mode -> {} ('{}')", state_string, source, mode_name, spec);
    };

    auto joystick_modes = cfg["gait_modes"];
    if (joystick_modes)
    {
        for (const auto& item : joystick_modes.as<std::map<std::string, std::string>>())
        {
            unitree::common::dsl::Parser parser(item.second);
            auto func = unitree::common::dsl::Compile(*parser.Parse());
            add(item.first, item.second, "joystick", [func]() -> bool {
                return func(FSMState::lowstate->joystick);
            });
        }
    }

    auto keyboard_modes = cfg["keyboard_gait_modes"];
    if (keyboard_modes && FSMState::keyboard)
    {
        for (const auto& item : keyboard_modes.as<std::map<std::string, std::string>>())
        {
            add(item.first, item.second, "keyboard", [key = item.second]() -> bool {
                return FSMState::keyboard && FSMState::keyboard->key() == key
                       && FSMState::keyboard->on_pressed;
            });
        }
    }
}

void State_BipedRL::enter()
{
    // The current pinned-gait_mode checkpoints (e.g. Go2-Biped-Phase1) only ever see
    // one mode for the entire episode, so entry mirrors that fixed mode instead of the
    // old resampling-command's "always starts quad" behavior.
    go2::GaitModeSelector::instance().set(default_gait_mode_);
    spdlog::info("Gait mode: {}", go2::gait_mode_name(default_gait_mode_));
    quad_tilt_limit_armed_ = false;

    // Before State_RLBase::enter(), which starts the policy thread that writes the rows.
    reset_diagnostics();
    open_telemetry();

    State_RLBase::enter();
}

void State_BipedRL::exit()
{
    // Joins the policy thread first: no further log_telemetry_row() call can be in
    // flight by the time the stream is closed.
    State_RLBase::exit();

    if (telemetry_log_.is_open())
    {
        telemetry_log_.close();
    }
    // No-op if the window already closed mid-run and wrote itself out.
    write_torque_capture();
    if (torque_writer_.joinable())
    {
        torque_writer_.join();
    }
    report_summary();
}

State_BipedRL::~State_BipedRL()
{
    if (torque_writer_.joinable())
    {
        torque_writer_.join();
    }
}

void State_BipedRL::run()
{
    auto& selector = go2::GaitModeSelector::instance();
    for (const auto& trigger : mode_triggers_)
    {
        if (trigger.triggered())
        {
            if (trigger.mode != selector.get())
            {
                selector.set(trigger.mode);
                quad_tilt_limit_armed_ = false;
                spdlog::info("Gait mode: {}", go2::gait_mode_name(trigger.mode));
            }
            break;
        }
    }

    // Runs before the FSM evaluates registered_checks on this tick, so fall_detected()
    // always sees an up-to-date arming decision.
    update_tilt_limit_arming();

    State_RLBase::run();

    // After the q targets are published, so tau_cmd pairs the position command actually
    // in force this tick with the joint state it acts on.
    sample_diagnostics();
}

void State_BipedRL::update_tilt_limit_arming()
{
    if (quad_tilt_limit_armed_
        || go2::GaitModeSelector::instance().get() != go2::GaitMode::kQuad)
    {
        return;
    }

    // Commanding quad from a biped stance only asks the policy to start lowering the
    // robot; it stays pitched well past tilt_limit_ for the seconds that takes, so
    // arming the strict limit on the command alone would report a fall on every
    // biped->quad switch. Wait for the pose itself to settle instead of guessing how
    // long the descent needs. Until then biped_tilt_limit_ stays in force, which still
    // catches a flip -- a fall part-way through the descent is indistinguishable from
    // the descent itself by tilt magnitude alone.
    const float arm_below = std::max(0.0f, tilt_limit_ - kQuadTiltArmMargin);
    if (!isaaclab::mdp::bad_orientation(env.get(), arm_below))
    {
        quad_tilt_limit_armed_ = true;
    }
}

bool State_BipedRL::fall_detected() const
{
    const bool is_quad = go2::GaitModeSelector::instance().get() == go2::GaitMode::kQuad;
    const bool strict = is_quad && quad_tilt_limit_armed_;
    return isaaclab::mdp::bad_orientation(env.get(), strict ? tilt_limit_ : biped_tilt_limit_);
}

/* -------------------------------------------------------------------------------- */
/* Diagnostics                                                                      */
/* -------------------------------------------------------------------------------- */

float State_BipedRL::tilt_deg() const
{
    // The same quantity bad_orientation() thresholds -- the angle between the robot's
    // own down axis and gravity -- but read straight from lowstate, so it is current at
    // the 1 kHz FSM rate rather than at the policy rate.
    const auto& quat = lowstate->msg_.imu_state().quaternion();
    const Eigen::Quaternionf root_quat(quat[0], quat[1], quat[2], quat[3]);
    const float grav_z = (root_quat.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f)).z();
    return std::acos(std::clamp(-grav_z, -1.0f, 1.0f)) * kRadToDeg;
}

void State_BipedRL::reset_diagnostics()
{
    log_stamp_ = timestamp_now();
    telemetry_step_ = 0;
    accumulated_pitch_deg_ = 0.0f;
    fsm_step_ = 0;
    torque_capture_.clear();
    if (torque_log_enabled_)
    {
        // Re-reserved because a previous entry's buffer was moved out to the writer
        // thread: without this, the 1 kHz push_back path would reallocate (and copy
        // megabytes) inside an FSM tick.
        torque_capture_.reserve(static_cast<size_t>(torque_log_s_ / kFsmDt) + 1);
    }
    torque_written_ = false;
    peak_tilt_deg_ = 0.0f;
    peak_tilt_time_s_ = 0.0f;
    peak_tau_ = 0.0f;
    peak_tau_joint_ = -1;
    peak_clamp_ = 0.0f;
    peak_clamp_joint_ = -1;
    peak_track_err_ = 0.0f;
    peak_track_err_joint_ = -1;
    peak_joint_vel_ = 0.0f;
}

void State_BipedRL::open_telemetry()
{
    if (!telemetry_enabled_)
    {
        return;
    }

    std::filesystem::path dir = log_dir_.empty() ? std::filesystem::current_path()
                                                 : std::filesystem::path(log_dir_);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto path = dir / (state_name_ + "_telemetry_" + log_stamp_ + ".csv");

    telemetry_log_.open(path, std::ios::out | std::ios::trunc);
    if (!telemetry_log_)
    {
        spdlog::warn("State_{}: could not open {} for telemetry", state_name_, path.string());
        return;
    }

    // step_ms is the previous step's measured loop time (it is stamped after this row is
    // written), so it lags one row -- enough to spot the control rate collapsing, not
    // meant for per-row attribution.
    telemetry_log_ << "t,gait_mode,tilt_deg,accumulated_pitch_deg,"
                   << "grav_x,grav_y,grav_z,gyro_x,gyro_y,gyro_z,"
                   << "cmd_vx,cmd_vy,cmd_wz,base_z,step_ms";
    // q_des is the action the policy asked for, q what the joint reached: their
    // difference is the whole PD-tracking story at kp=25, and tau_est says whether the
    // motor was even trying. Named by real joint, see kSdkJointNames.
    const char* fields[4] = {"q", "dq", "q_des", "tau_est"};
    for (int f = 0; f < 4; ++f)
    {
        for (int i = 0; i < 12; ++i)
        {
            const int sdk_index = static_cast<int>(env->robot->data.joint_ids_map[i]);
            telemetry_log_ << "," << kSdkJointNames[sdk_index] << "_" << fields[f];
        }
    }
    telemetry_log_ << "\n";

    spdlog::info("State_{}: writing 50Hz telemetry to {}", state_name_, path.string());
}

void State_BipedRL::on_policy_step()
{
    log_telemetry_row();
}

void State_BipedRL::log_telemetry_row()
{
    if (!telemetry_log_.is_open())
    {
        return;
    }

    auto& data = env->robot->data;

    // Integrated body-frame pitch rate. Redundant with tilt_deg for a pure sagittal
    // rise, but it is signed -- so a front stance (nose down) and a hind stance (nose
    // up) are told apart, and a stance that starts rotating the wrong way shows up
    // immediately.
    accumulated_pitch_deg_ += data.root_ang_vel_b.y() * env->step_dt * kRadToDeg;

    const auto command = go2::keyboard_velocity_command(env.get());

    // Copied under the lowstate lock (the FSM thread is publishing into it at 1 kHz),
    // then formatted outside it.
    float q[12], dq[12], tau[12];
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        for (int i = 0; i < 12; ++i)
        {
            const int sdk_index = static_cast<int>(data.joint_ids_map[i]);
            const auto& motor = lowstate->msg_.motor_state()[sdk_index];
            q[i] = motor.q();
            dq[i] = motor.dq();
            tau[i] = motor.tau_est();
        }
    }
    const auto q_des = env->action_manager->processed_actions();

    telemetry_log_ << (telemetry_step_ * env->step_dt) << ","
                   << static_cast<int>(go2::GaitModeSelector::instance().get()) << ","
                   << tilt_deg() << "," << accumulated_pitch_deg_ << ","
                   << data.projected_gravity_b.x() << "," << data.projected_gravity_b.y() << ","
                   << data.projected_gravity_b.z() << ","
                   << data.root_ang_vel_b.x() << "," << data.root_ang_vel_b.y() << ","
                   << data.root_ang_vel_b.z();
    for (int i = 0; i < 3; ++i)
    {
        telemetry_log_ << "," << (i < static_cast<int>(command.size()) ? command[i] : 0.0f);
    }
    telemetry_log_ << "," << (base_height_ ? base_height_->msg_.position()[2] : 0.0f) << ","
                   << policy_step_last_ms_.load();

    for (int i = 0; i < 12; ++i) telemetry_log_ << "," << q[i];
    for (int i = 0; i < 12; ++i) telemetry_log_ << "," << dq[i];
    for (int i = 0; i < 12; ++i)
    {
        telemetry_log_ << "," << (i < static_cast<int>(q_des.size()) ? q_des[i] : 0.0f);
    }
    for (int i = 0; i < 12; ++i) telemetry_log_ << "," << tau[i];
    telemetry_log_ << "\n";
    // Flushed per row: an attempt that ends in a fall is usually killed by hand, and a
    // buffered tail would lose exactly the part being investigated. The cost shows up in
    // the step_ms column, so it cannot hide.
    telemetry_log_.flush();

    telemetry_step_ += 1;
}

void State_BipedRL::sample_diagnostics()
{
    fsm_step_ += 1;
    const float t = fsm_step_ * kFsmDt;

    const float tilt = tilt_deg();
    if (tilt > peak_tilt_deg_)
    {
        peak_tilt_deg_ = tilt;
        peak_tilt_time_s_ = t;
    }

    const bool capture = torque_log_enabled_ && !torque_written_ && t <= torque_log_s_;
    TorqueSample s;
    if (capture)
    {
        s.step = fsm_step_;
        s.tilt_deg = tilt;
        s.base_z = base_height_ ? static_cast<float>(base_height_->msg_.position()[2]) : 0.0f;
    }

    for (int i = 0; i < 12; ++i)
    {
        const int sdk_index = static_cast<int>(env->robot->data.joint_ids_map[i]);
        const auto& motor = lowstate->msg_.motor_state()[sdk_index];
        const auto& cmd = lowcmd->msg_.motor_cmd()[sdk_index];

        // The expression the motor firmware applies, and the same one the MuJoCo bridge
        // evaluates (unitree_sdk2_bridge.h), so the two traces are directly comparable.
        const float tau_cmd =
            cmd.tau() + cmd.kp() * (cmd.q() - motor.q()) + cmd.kd() * (cmd.dq() - motor.dq());
        const float tau_app = motor.tau_est();
        const float track_err = std::fabs(cmd.q() - motor.q());

        if (std::fabs(tau_app) > peak_tau_)
        {
            peak_tau_ = std::fabs(tau_app);
            peak_tau_joint_ = sdk_index;
        }
        const float clamped = std::fabs(tau_cmd) - std::fabs(tau_app);
        if (clamped > peak_clamp_)
        {
            peak_clamp_ = clamped;
            peak_clamp_joint_ = sdk_index;
        }
        if (track_err > peak_track_err_)
        {
            peak_track_err_ = track_err;
            peak_track_err_joint_ = sdk_index;
        }
        peak_joint_vel_ = std::max(peak_joint_vel_, std::fabs(motor.dq()));

        if (capture)
        {
            s.q[i] = motor.q();
            s.dq[i] = motor.dq();
            s.q_des[i] = cmd.q();
            s.tau_cmd[i] = tau_cmd;
            s.tau_app[i] = tau_app;
        }
    }

    if (capture)
    {
        torque_capture_.push_back(s);
        if (t >= torque_log_s_)
        {
            // Written as soon as the window closes rather than on exit, so the file is
            // already on disk if the attempt has to be killed by hand.
            write_torque_capture();
        }
    }
}

void State_BipedRL::write_torque_capture()
{
    if (torque_written_ || torque_capture_.empty())
    {
        return;
    }
    torque_written_ = true;

    std::filesystem::path dir = log_dir_.empty() ? std::filesystem::current_path()
                                                 : std::filesystem::path(log_dir_);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto path = dir / (state_name_ + "_torque_" + log_stamp_ + ".csv");

    std::string header = "t,tilt_deg,base_z";
    const char* fields[5] = {"q", "dq", "q_des", "tau_cmd", "tau_app"};
    for (int f = 0; f < 5; ++f)
    {
        for (int i = 0; i < 12; ++i)
        {
            const int sdk_index = static_cast<int>(env->robot->data.joint_ids_map[i]);
            header += ",";
            header += kSdkJointNames[sdk_index];
            header += "_";
            header += fields[f];
        }
    }
    header += "\n";

    // Hand the serialization to a worker: this runs on the 1 kHz FSM thread, and at the
    // moment the window closes the robot is still standing on two legs.
    if (torque_writer_.joinable())
    {
        torque_writer_.join();
    }
    torque_writer_ = std::thread(
        &State_BipedRL::write_torque_csv, path.string(), std::move(header),
        std::move(torque_capture_));
    torque_capture_.clear();  // moved-from; back to a defined empty state
}

void State_BipedRL::write_torque_csv(
    std::string path, std::string header, std::vector<TorqueSample> samples)
{
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out)
    {
        spdlog::warn("could not open {} for torque capture", path);
        return;
    }

    out << header;
    for (const auto& s : samples)
    {
        out << s.step * kFsmDt << "," << s.tilt_deg << "," << s.base_z;
        for (int i = 0; i < 12; ++i) out << "," << s.q[i];
        for (int i = 0; i < 12; ++i) out << "," << s.dq[i];
        for (int i = 0; i < 12; ++i) out << "," << s.q_des[i];
        for (int i = 0; i < 12; ++i) out << "," << s.tau_cmd[i];
        for (int i = 0; i < 12; ++i) out << "," << s.tau_app[i];
        out << "\n";
    }
    out.close();

    spdlog::info("wrote {} ({} rows, 0.00s .. {:.2f}s from entry)",
        path, samples.size(), samples.empty() ? 0.0f : samples.back().step * kFsmDt);
}

void State_BipedRL::report_summary() const
{
    const auto joint_name = [](int sdk_index) {
        return (sdk_index >= 0 && sdk_index < 12) ? kSdkJointNames[sdk_index] : "n/a";
    };

    // peak tilt is the "did it get up at all" number: a biped stance holds the base
    // 70-90 deg off flat (Isaac Lab's tilt_reward plateaus around 73 deg for these
    // checkpoints), so a peak far below that means the rise never happened -- and the
    // torque/tracking peaks next to it say whether the policy even asked for it.
    spdlog::info(
        "State_{}: {:.1f}s active, peak tilt {:.1f} deg @{:.2f}s (biped stance is ~70-90 deg) -- "
        "peak |tau| {:.1f} Nm ({}), peak clamped torque {:.1f} Nm ({}), "
        "peak |q_des - q| {:.3f} rad ({}), peak |dq| {:.1f} rad/s -- "
        "policy step max {:.1f} ms of {:.1f} ms budget, {} overrun(s)",
        state_name_, fsm_step_ * kFsmDt, peak_tilt_deg_, peak_tilt_time_s_,
        peak_tau_, joint_name(peak_tau_joint_),
        peak_clamp_, joint_name(peak_clamp_joint_),
        peak_track_err_, joint_name(peak_track_err_joint_),
        peak_joint_vel_,
        policy_step_max_ms_.load(), env->step_dt * 1e3f, policy_overrun_count_.load());
}
