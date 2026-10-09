// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#include "State_Flip.h"
#include "FlipLogger.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <algorithm>
#include <cctype>
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
    return std::vector<float>{cubed / (1.0f + cubed)};
}

} // namespace mdp
} // namespace isaaclab

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
    action_map_ = std::make_unique<JointActionMap>(env->cfg);

    command_ = std::make_shared<FlipCommand>();
    if (cfg["time_scale"])            command_->time_scale = cfg["time_scale"].as<float>();
    if (cfg["command_duration_s"])    command_->command_duration_s = cfg["command_duration_s"].as<float>();
    if (cfg["rearm_delay_s"])         command_->rearm_delay_s = cfg["rearm_delay_s"].as<float>();
    if (cfg["fall_check_delay_s"])    fall_check_delay_s_ = cfg["fall_check_delay_s"].as<float>();
    if (cfg["fall_check_hold_s"])     fall_check_hold_s_ = cfg["fall_check_hold_s"].as<float>();
    if (cfg["bad_orientation_limit"]) bad_orientation_limit_ = cfg["bad_orientation_limit"].as<float>();
    if (auto brake = cfg["wheel_brake"])
    {
        if (brake["enabled"])   wheel_brake_enabled_ = brake["enabled"].as<bool>();
        if (brake["kp"])        wheel_brake_kp_ = brake["kp"].as<float>();
        if (brake["kd"])        wheel_brake_kd_ = brake["kd"].as<float>();
        if (brake["release_s"]) wheel_release_s_ = brake["release_s"].as<float>();
    }
    // Same keys as the Go2 controller on feat/jump, plus log_dir. Telemetry is on unless
    // disabled; the 1 kHz torque capture is opt-in (one CSV per motion).
    FlipLogger::Config log_cfg;
    if (cfg["log_dir"])           log_cfg.log_dir = cfg["log_dir"].as<std::string>();
    if (cfg["telemetry_log"])     log_cfg.telemetry = cfg["telemetry_log"].as<bool>();
    if (cfg["torque_log"])        log_cfg.torque_log = cfg["torque_log"].as<bool>();
    if (cfg["torque_log_pre_s"])  log_cfg.torque_pre_s = cfg["torque_log_pre_s"].as<float>();
    if (cfg["torque_log_post_s"]) log_cfg.torque_post_s = cfg["torque_log_post_s"].as<float>();
    logger_ = std::make_unique<FlipLogger>(log_cfg, FSMState::lowstate, FSMState::lowcmd.get());

    spdlog::info(
        "State_{}: wheel brake {} (kp {:.1f}, kd {:.1f}; released for {:.2f}s from each trigger)",
        state_string, wheel_brake_enabled_ ? "on" : "off", wheel_brake_kp_, wheel_brake_kd_, wheel_release_s_);

    // Every target is given explicitly in config.yaml: they are policy observations, so
    // they must be exactly what the policy was trained with -- including the height a flip
    // carries alongside its rotation.
    for (const auto & entry : cfg["motions"])
    {
        MotionPreset preset;
        preset.key = entry["key"].as<std::string>();
        preset.name = entry["motion"] ? entry["motion"].as<std::string>() : "";
        preset.target_height = entry["target_height"] ? entry["target_height"].as<float>() : 0.0f;
        preset.target_pitch_turns = entry["target_pitch_turns"] ? entry["target_pitch_turns"].as<float>() : 0.0f;
        preset.target_roll_turns = entry["target_roll_turns"] ? entry["target_roll_turns"].as<float>() : 0.0f;
        motions_.push_back(preset);
        spdlog::info(
            "State_{}: key '{}' -> {} [h={:.2f} pitch={:.2f} roll={:.2f}]",
            state_string, preset.key, preset.name,
            preset.target_height, preset.target_pitch_turns, preset.target_roll_turns);
    }
    if (motions_.empty())
    {
        throw std::runtime_error("State_" + state_string + ": no `motions:` configured");
    }

    // A fall *after* the motion window sends the robot to Passive. The check is gated
    // until command_duration_s + fall_check_delay_s after the trigger, so it never trips on
    // the expected in-air tilt or the landing settle.
    this->registered_checks.emplace_back(
        std::make_pair(
            [this]() -> bool
            {
                if (!command_ || command_->trigger_step < 0
                    || command_->elapsed() < command_->command_duration_s + fall_check_delay_s_
                    || !isaaclab::mdp::bad_orientation(env.get(), bad_orientation_limit_))
                {
                    bad_orientation_latched_ = false;
                    return false;
                }
                // Tilted past the limit -- require it to stay that way before giving up on
                // the robot (this runs at 1 kHz, and a touchdown shock swings the fused IMU
                // orientation for a few milliseconds).
                const auto now = std::chrono::steady_clock::now();
                if (!bad_orientation_latched_)
                {
                    bad_orientation_latched_ = true;
                    bad_orientation_since_ = now;
                    return false;
                }
                const float held_s = std::chrono::duration<float>(now - bad_orientation_since_).count();
                if (held_s < fall_check_hold_s_)
                {
                    return false;
                }
                spdlog::warn("State_Flip: fall detected {:.2f}s after trigger -> Passive. {}",
                             command_->elapsed(), logger_->impact_summary());
                return true;
            },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_Flip::enter()
{
    // deploy.yaml stores stiffness/damping already in SDK motor order (export_deploy_cfg).
    for (size_t i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        auto & motor = lowcmd->msg_.motor_cmd()[i];
        motor.kp() = env->robot->data.joint_stiffness[i];
        motor.kd() = env->robot->data.joint_damping[i];
        motor.dq() = 0;
        motor.tau() = 0;
    }

    command_->step_dt = env->step_dt;
    command_->reset();
    command = command_;
    bad_orientation_latched_ = false;
    wheel_braking_ = false;
    logger_->on_enter(command_.get(), env->step_dt, command_->command_duration_s + fall_check_delay_s_);

    env->robot->update();

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
            // Advance the command clock *before* computing observations so jump_command /
            // jump_time reflect the current step.
            command_->step();
            env->step();
            logger_->on_policy_step(
                std::chrono::duration<float, std::milli>(clock::now() - work_start).count(), wheel_braking_);

            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_Flip::run()
{
    if (keyboard && keyboard->on_pressed)
    {
        const std::string key = keyboard->key();
        for (const auto & m : motions_)
        {
            if (m.key == key)
            {
                spdlog::info("State_Flip: {} requested", m.name);
                command_->request(m.target_height, m.target_pitch_turns, m.target_roll_turns);
                break;
            }
        }
    }

    action_map_->write(env->action_manager->processed_actions(), *lowcmd);
    apply_wheel_brake();
    logger_->on_fsm_tick();
}

void State_Flip::exit()
{
    policy_thread_running = false;
    if (policy_thread.joinable())
    {
        policy_thread.join();
    }
    logger_->on_exit();
}

State_Flip::~State_Flip() = default;

bool State_Flip::wheels_released() const
{
    return command_->trigger_step >= 0 && command_->elapsed() < wheel_release_s_;
}

void State_Flip::apply_wheel_brake()
{
    const auto & wheel_ids = action_map_->vel_motor_ids();
    if (!wheel_brake_enabled_ || wheels_released())
    {
        if (wheel_braking_)
        {
            // Hand the wheels back to the policy with the trained gains (kp 0, kd 0.5).
            for (int id : wheel_ids)
            {
                lowcmd->msg_.motor_cmd()[id].kp() = env->robot->data.joint_stiffness[id];
                lowcmd->msg_.motor_cmd()[id].kd() = env->robot->data.joint_damping[id];
            }
            wheel_braking_ = false;
        }
        return;
    }

    if (!wheel_braking_)
    {
        // Lock where the wheels are now, so engaging the brake does not snap them back.
        wheel_lock_q_.clear();
        for (int id : wheel_ids)
        {
            wheel_lock_q_.push_back(lowstate->msg_.motor_state()[id].q());
        }
        wheel_braking_ = true;
    }
    for (size_t i = 0; i < wheel_ids.size(); ++i)
    {
        auto & motor = lowcmd->msg_.motor_cmd()[wheel_ids[i]];
        motor.q() = wheel_lock_q_[i];
        motor.dq() = 0.0f;
        motor.kp() = wheel_brake_kp_;
        motor.kd() = wheel_brake_kd_;
    }
}
