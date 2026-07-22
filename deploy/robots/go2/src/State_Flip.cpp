// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#include "State_Flip.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

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
    // check is gated until command_duration_s + a small settle delay has
    // elapsed since the trigger (Phase 3 disables bad_orientation entirely).
    this->registered_checks.emplace_back(
        std::make_pair(
            [this]() -> bool
            {
                // No motion issued yet (still just standing) -> nothing to guard.
                if (!command_ || command_->trigger_step < 0)
                {
                    return false;
                }
                // Inside the (expected upside-down) motion window -> don't trip.
                const float since_trigger = command_->elapsed();
                if (since_trigger < command_->command_duration_s + fall_check_delay_s_)
                {
                    return false;
                }
                return isaaclab::mdp::bad_orientation(env.get(), bad_orientation_limit_);
            },
            FSMStringMap.right.at("Passive")
        )
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

    env->robot->update();

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
            // Advance the command clock *before* computing observations so the
            // jump_command / jump_time terms reflect the current step.
            command_->step();
            env->step();

            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_Flip::run()
{
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
