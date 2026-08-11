// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#include "State_Flip.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <spdlog/fmt/fmt.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>

std::shared_ptr<State_Flip::FlipCommand> State_Flip::command = nullptr;

namespace isaaclab
{
namespace mdp
{

// jump_command : [enabled, target_height, target_pitch_turns, target_roll_turns]
// Mirrors JumpCommand.command (commands.py). Targets are zeroed while disabled.
REGISTER_OBSERVATION(jump_command)
{
    (void)env;
    (void)params;
    auto & cmd = State_Flip::command;

    std::vector<float> obs(4, 0.0f);
    if (cmd)
    {
        const float enabled = cmd->enabled ? 1.0f : 0.0f;
        obs[0] = enabled;
        obs[1] = cmd->target_height * enabled;
        obs[2] = cmd->target_pitch_turns * enabled;
        obs[3] = cmd->target_roll_turns * enabled;
    }
    return obs;
}

// jump_time : bounded cubic time encoding since the command rising edge.
// Mirrors jump_time_encoding (observations.py): (t/s)^3 / (1 + (t/s)^3).
REGISTER_OBSERVATION(jump_time)
{
    (void)env;
    auto & cmd = State_Flip::command;

    float t = 0.0f;
    float time_scale = 1.0f;
    if (cmd)
    {
        time_scale = cmd->time_scale;
        t = cmd->time_since_trigger();
    }
    // Allow deploy.yaml to override time_scale via params (matches the python
    // ObsTerm signature default of 1.0 when unset).
    if (params["time_scale"] && !params["time_scale"].IsNull())
    {
        time_scale = params["time_scale"].as<float>();
    }
    if (time_scale <= 0.0f)
    {
        time_scale = 1.0f;
    }

    const float scaled = t / time_scale;
    const float cubed = scaled * scaled * scaled;
    const float encoded = cubed / (1.0f + cubed);
    return std::vector<float>{encoded};
}

} // namespace mdp
} // namespace isaaclab

namespace
{
std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}
} // namespace

State_Flip::State_Flip(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    command_ = std::make_shared<FlipCommand>();

    // --- timing (defaults mirror JumpCommandCfg) ---
    if (cfg["time_scale"])          command_->time_scale = cfg["time_scale"].as<float>();
    if (cfg["command_duration_s"])  command_->command_duration_s = cfg["command_duration_s"].as<float>();
    if (cfg["trigger_delay_s"])     command_->trigger_delay_s = cfg["trigger_delay_s"].as<float>();
    if (cfg["rearm_delay_s"])       command_->rearm_delay_s = cfg["rearm_delay_s"].as<float>();
    if (cfg["fall_check_delay_s"])  fall_check_delay_s_ = cfg["fall_check_delay_s"].as<float>();
    if (cfg["fall_check_hold_s"])   fall_check_hold_s_ = cfg["fall_check_hold_s"].as<float>();
    if (cfg["bad_orientation_limit"]) bad_orientation_limit_ = cfg["bad_orientation_limit"].as<float>();

    // Fill per-motion targets from a `motion` string / explicit targets.
    auto apply_targets = [&](YAML::Node node, const std::string & motion,
                             float & h, float & p, float & r)
    {
        h = node["target_height"] ? node["target_height"].as<float>() : 0.0f;
        p = node["target_pitch_turns"] ? node["target_pitch_turns"].as<float>() : 0.0f;
        r = node["target_roll_turns"] ? node["target_roll_turns"].as<float>() : 0.0f;
        if (h == 0.0f && p == 0.0f && r == 0.0f)
        {
            if (motion == "jump")          h = 0.20f;
            else if (motion == "backflip") p = -1.0f;
            else if (motion == "sideflip") r = 1.0f;
        }
    };

    if (cfg["motions"] && cfg["motions"].IsSequence())
    {
        // --- manual / Dynamic mode: keyboard-selectable, re-armable motions ---
        manual_mode_ = true;
        command_->manual_mode = true;
        command_->use_auto_trigger = false;

        for (const auto & entry : cfg["motions"])
        {
            MotionPreset preset;
            preset.key = entry["key"].as<std::string>();
            preset.name = entry["motion"] ? to_lower(entry["motion"].as<std::string>()) : "";
            apply_targets(entry, preset.name, preset.target_height,
                          preset.target_pitch_turns, preset.target_roll_turns);
            motions_.push_back(preset);
            spdlog::info(
                "State_{}: motion key '{}' -> {} [h={:.2f} pitch={:.2f} roll={:.2f}]",
                state_string, preset.key,
                preset.name.empty() ? "(targets)" : preset.name,
                preset.target_height, preset.target_pitch_turns, preset.target_roll_turns);
        }
        spdlog::info(
            "State_{}: manual multi-motion mode (press keys to fire, re-arm {:.2f}s, duration {:.2f}s)",
            state_string, command_->rearm_delay_s, command_->command_duration_s);
    }
    else
    {
        // --- single-shot mode: one motion, auto-fired (or fired by trigger_key) ---
        std::string motion = cfg["motion"] ? to_lower(cfg["motion"].as<std::string>()) : "";
        apply_targets(cfg, motion, command_->target_height,
                      command_->target_pitch_turns, command_->target_roll_turns);

        if (cfg["trigger_key"])
        {
            trigger_key_ = cfg["trigger_key"].as<std::string>();
            command_->use_auto_trigger = false;
        }

        spdlog::info(
            "State_{}: motion='{}' targets[h={:.2f} pitch={:.2f} roll={:.2f}] "
            "trigger={} duration={:.2f}s",
            state_string,
            motion.empty() ? "(from targets)" : motion,
            command_->target_height,
            command_->target_pitch_turns,
            command_->target_roll_turns,
            trigger_key_.empty() ? ("auto@" + std::to_string(command_->trigger_delay_s) + "s") : ("key='" + trigger_key_ + "'"),
            command_->command_duration_s);
    }

    // A fall *after* the motion window sends the robot to Passive. We must not
    // trip on the (expected) large in-air tilt during the flip itself, so the
    // check is gated until command_duration_s + a settle delay has elapsed since
    // the trigger (Phase 2 disables bad_orientation entirely).
    this->registered_checks.emplace_back(
        [this]() -> bool
        {
            // No motion issued yet (still just standing) -> nothing to guard.
            if (!command_ || command_->trigger_step < 0)
            {
                bad_orientation_latched_ = false;
                return false;
            }
            // Inside the (expected upside-down) motion window -> don't trip.
            const float since_trigger = command_->elapsed();
            if (since_trigger < command_->command_duration_s + fall_check_delay_s_)
            {
                bad_orientation_latched_ = false;
                return false;
            }
            if (!isaaclab::mdp::bad_orientation(env.get(), bad_orientation_limit_))
            {
                bad_orientation_latched_ = false;
                return false;
            }

            // Tilted past the limit -- but require it to *stay* that way before
            // giving up on the robot. bad_orientation reads the real IMU's fused
            // orientation, whose gravity direction swings for a few milliseconds on
            // a hard landing impact, and this runs at the 1 kHz FSM rate, so a
            // single glitched sample would otherwise drop a perfectly good landing
            // to Passive. Simulation never sees this: its orientation is ground
            // truth (and Phase 2 has no bad_orientation termination).
            const auto now = std::chrono::steady_clock::now();
            if (!bad_orientation_latched_)
            {
                bad_orientation_latched_ = true;
                bad_orientation_since_ = now;
                return false;
            }
            const float held_s =
                std::chrono::duration<float>(now - bad_orientation_since_).count();
            if (held_s < fall_check_hold_s_)
            {
                return false;
            }

            spdlog::warn(
                "State_Flip: fall detected {:.2f}s after trigger "
                "(tilt {:.1f} deg > limit {:.1f} deg, held {:.3f}s). {}",
                since_trigger,
                tilt_deg(),
                bad_orientation_limit_ * 180.0f / static_cast<float>(M_PI),
                held_s,
                impact_summary());
            return true;
        },
        FSMStringMap.right.at("Passive"),
        "fall(bad_orientation)"
    );
}

void State_Flip::enter()
{
    // set gain
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    command_->step_dt = env->step_dt;
    command_->reset();
    command = command_; // expose to observation terms
    bad_orientation_latched_ = false;

    env->robot->update();

    // --- open diagnostic telemetry log ---
    telemetry_step_ = 0;
    accumulated_pitch_deg_ = 0.0f;
    telemetry_log_.open("telemetry.csv", std::ios::out | std::ios::trunc);
    telemetry_log_ << "t,enabled,elapsed_since_trigger,accumulated_pitch_deg,"
                   << "grav_x,grav_y,grav_z,tilt_deg,"
                   << "accel_x,accel_y,accel_z,"
                   << "FL_hip_vel,FR_hip_vel,RL_hip_vel,RR_hip_vel,"
                   << "FL_thigh_vel,FR_thigh_vel,RL_thigh_vel,RR_thigh_vel,"
                   << "FL_calf_vel,FR_calf_vel,RL_calf_vel,RR_calf_vel,"
                   << "FL_hip_tau,FR_hip_tau,RL_hip_tau,RR_hip_tau,"
                   << "FL_thigh_tau,FR_thigh_tau,RL_thigh_tau,RR_thigh_tau,"
                   << "FL_calf_tau,FR_calf_tau,RL_calf_tau,RR_calf_tau\n";

    // Start policy thread
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        auto sleepTill = clock::now() + dt;
        command_->reset();
        env->reset();

        while (policy_thread_running)
        {
            const auto work_start = clock::now();

            // Advance the command clock *before* computing observations so the
            // jump_command / jump_time terms reflect the current step.
            command_->step();
            env->step();
            log_telemetry_row();

            const float work_ms =
                std::chrono::duration<float, std::milli>(clock::now() - work_start).count();
            if (work_ms > policy_step_max_ms_.load())
            {
                policy_step_max_ms_.store(work_ms);
            }
            if (work_ms > env->step_dt * 1e3f)
            {
                policy_overrun_count_.fetch_add(1);
            }

            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_Flip::log_telemetry_row()
{
    // Mirrors the training-side accumulated_pitch computation (commands.py) so this
    // number is directly comparable to IsaacLab's Metrics/jump/max_height play-mode
    // trace, rather than relying on visual observation alone.
    const float pitch_rate = env->robot->data.root_ang_vel_b.y();
    accumulated_pitch_deg_ += pitch_rate * env->step_dt * 180.0f / static_cast<float>(M_PI);

    telemetry_log_ << (telemetry_step_ * env->step_dt) << ","
                   << (command_->enabled ? 1 : 0) << ","
                   << command_->elapsed() << ","
                   << accumulated_pitch_deg_ << ","
                   << env->robot->data.projected_gravity_b.x() << ","
                   << env->robot->data.projected_gravity_b.y() << ","
                   << env->robot->data.projected_gravity_b.z() << ","
                   << tilt_deg();

    // Raw IMU acceleration: the landing shock that perturbs the fused orientation
    // this state's fall guard reads, so the two are directly comparable per row.
    const auto & accel = lowstate->msg_.imu_state().accelerometer();
    for (int i = 0; i < 3; ++i)
    {
        telemetry_log_ << "," << accel[i];
    }

    for (int i = 0; i < 12; ++i)
    {
        telemetry_log_ << "," << env->robot->data.joint_vel[i];
    }
    for (int i = 0; i < 12; ++i)
    {
        const int sdk_index = static_cast<int>(env->robot->data.joint_ids_map[i]);
        telemetry_log_ << "," << lowstate->msg_.motor_state()[sdk_index].tau_est();
    }
    telemetry_log_ << "\n";
    telemetry_log_.flush();

    telemetry_step_ += 1;
}

float State_Flip::tilt_deg() const
{
    // The same quantity bad_orientation() thresholds -- the angle between the
    // robot's own down axis and gravity, from the IMU's fused orientation -- but
    // read straight from lowstate so it is current at the 1 kHz FSM rate rather
    // than at the policy rate.
    const auto & quat = lowstate->msg_.imu_state().quaternion();
    const Eigen::Quaternionf root_quat(quat[0], quat[1], quat[2], quat[3]);
    const float grav_z = (root_quat.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f)).z();
    return std::acos(std::clamp(-grav_z, -1.0f, 1.0f)) * 180.0f / static_cast<float>(M_PI);
}

std::string State_Flip::impact_summary() const
{
    return fmt::format(
        "peak |accel| {:.1f} m/s^2 @{:.2f}s, peak |tau| {:.1f} Nm (motor {}), "
        "peak |dq| {:.1f} rad/s, tilt at guard arm {:.1f} deg, "
        "peak tilt after arm {:.1f} deg @{:.2f}s (limit {:.1f} deg), "
        "policy step max {:.1f} ms of {:.1f} ms budget, {} overrun(s)",
        peak_accel_, peak_accel_time_s_, peak_tau_, peak_tau_motor_,
        peak_joint_vel_, tilt_at_arm_deg_, peak_tilt_deg_, peak_tilt_time_s_,
        bad_orientation_limit_ * 180.0f / static_cast<float>(M_PI),
        policy_step_max_ms_.load(), env->step_dt * 1e3f,
        policy_overrun_count_.load());
}

void State_Flip::report_impact() const
{
    spdlog::info("State_Flip: motion landed -- {}", impact_summary());
}

void State_Flip::sample_diagnostics()
{
    if (!command_ || command_->trigger_step < 0)
    {
        return;
    }

    // A new motion was fired. Flush the previous one's numbers first, since
    // chaining motions faster than the report delay would otherwise lose them.
    if (command_->trigger_step != diag_trigger_step_)
    {
        if (diag_trigger_step_ >= 0 && !diag_reported_)
        {
            report_impact();
        }
        diag_trigger_step_ = command_->trigger_step;
        diag_reported_ = false;
        peak_accel_ = 0.0f;
        peak_accel_time_s_ = 0.0f;
        peak_tau_ = 0.0f;
        peak_tau_motor_ = -1;
        peak_joint_vel_ = 0.0f;
        tilt_at_arm_deg_ = -1.0f;
        peak_tilt_deg_ = 0.0f;
        peak_tilt_time_s_ = 0.0f;
        policy_overrun_count_.store(0);
        policy_step_max_ms_.store(0.0f);
    }

    const float since_trigger = command_->elapsed();

    const auto & accel = lowstate->msg_.imu_state().accelerometer();
    const float accel_norm =
        std::sqrt(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]);
    if (accel_norm > peak_accel_)
    {
        peak_accel_ = accel_norm;
        peak_accel_time_s_ = since_trigger;
    }

    for (int i = 0; i < 12; ++i)
    {
        const int sdk_index = static_cast<int>(env->robot->data.joint_ids_map[i]);
        const auto & motor = lowstate->msg_.motor_state()[sdk_index];
        const float tau = std::fabs(motor.tau_est());
        if (tau > peak_tau_)
        {
            peak_tau_ = tau;
            peak_tau_motor_ = sdk_index;
        }
        peak_joint_vel_ = std::max(peak_joint_vel_, std::fabs(motor.dq()));
    }

    const float arm_time = command_->command_duration_s + fall_check_delay_s_;
    if (since_trigger >= arm_time)
    {
        const float tilt = tilt_deg();
        if (tilt_at_arm_deg_ < 0.0f)
        {
            tilt_at_arm_deg_ = tilt;
        }
        if (tilt > peak_tilt_deg_)
        {
            peak_tilt_deg_ = tilt;
            peak_tilt_time_s_ = since_trigger;
        }
    }

    // Report once the guard has been armed long enough for its numbers to mean
    // something, so a motion that lands *without* tripping still gets logged --
    // that is what makes a backflip / sideflip comparison possible.
    if (!diag_reported_ && since_trigger >= arm_time + 0.5f)
    {
        report_impact();
        diag_reported_ = true;
    }
}

void State_Flip::run()
{
    sample_diagnostics();

    if (keyboard)
    {
        if (manual_mode_)
        {
            // Dynamic mode: fire the matching motion on a fresh key press.
            // Edge-triggered (on_pressed) so a held key fires only once.
            if (keyboard->on_pressed)
            {
                const std::string key = keyboard->key();
                for (const auto & m : motions_)
                {
                    if (m.key == key)
                    {
                        command_->request(m.target_height, m.target_pitch_turns, m.target_roll_turns);
                        break;
                    }
                }
            }
        }
        else if (!trigger_key_.empty() && keyboard->pressed(trigger_key_))
        {
            // Single-shot mode: optional manual trigger for the one motion.
            command_->trigger_requested.store(true);
        }
    }

    auto action = env->action_manager->processed_actions();
    for (int i(0); i < env->robot->data.joint_ids_map.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
