#include "State_Multitask.h"

#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <algorithm>
#include <cctype>

namespace
{
std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// Same defaults State_Flip applies when a motion names itself but gives no explicit targets, so a
// config entry means the same thing in either state.
void apply_targets(YAML::Node node, const std::string & motion, float & h, float & p, float & r)
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
}
}  // namespace

State_Multitask::State_Multitask(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    command_ = std::make_shared<State_Flip::FlipCommand>();
    // Always on-demand here: the robot is driving, and a move is an interruption the operator asks
    // for. Auto-trigger belongs to the acrobatics task, where one flip *is* the episode.
    command_->manual_mode = true;
    command_->use_auto_trigger = false;

    // command_duration_s has to match the value the policy was trained with. It is not just a
    // deploy-side timeout: it sets how long the `enabled` flag the network reads stays high, and in
    // the merged policy the gate's routing prior keys off that same flag.
    if (cfg["time_scale"])            command_->time_scale = cfg["time_scale"].as<float>();
    if (cfg["command_duration_s"])    command_->command_duration_s = cfg["command_duration_s"].as<float>();
    if (cfg["rearm_delay_s"])         command_->rearm_delay_s = cfg["rearm_delay_s"].as<float>();
    if (cfg["fall_check_delay_s"])    fall_check_delay_s_ = cfg["fall_check_delay_s"].as<float>();
    if (cfg["fall_check_hold_s"])     fall_check_hold_s_ = cfg["fall_check_hold_s"].as<float>();
    if (cfg["bad_orientation_limit"]) bad_orientation_limit_ = cfg["bad_orientation_limit"].as<float>();

    if (cfg["motions"] && cfg["motions"].IsSequence())
    {
        for (const auto & entry : cfg["motions"])
        {
            State_Flip::MotionPreset preset;
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
    }
    else
    {
        spdlog::warn(
            "State_{}: no `motions:` list configured -- the robot will drive but no acrobatic move "
            "can be fired.", state_string);
    }

    spdlog::info(
        "State_{}: drive + on-demand motions (duration {:.2f}s, re-arm {:.2f}s, fall guard arms "
        "{:.2f}s after a trigger)",
        state_string, command_->command_duration_s, command_->rearm_delay_s,
        command_->command_duration_s + fall_check_delay_s_);

    // A fall sends the robot to Passive. Unlike State_Flip this state is guarding a robot that is
    // usually *running*, so the guard is live by default and suppressed only around a commanded
    // move -- the same shape as the training environment, where bad_orientation is gated off inside
    // the acrobatics window and active everywhere else.
    this->registered_checks.emplace_back(
        [this]() -> bool
        {
            if (!command_)
            {
                return false;
            }

            // Inside the window the robot is upside down because it was told to be.
            if (command_->trigger_step >= 0)
            {
                const float since_trigger = command_->elapsed();
                if (since_trigger < command_->command_duration_s + fall_check_delay_s_)
                {
                    bad_orientation_latched_ = false;
                    return false;
                }
            }

            if (!isaaclab::mdp::bad_orientation(env.get(), bad_orientation_limit_))
            {
                bad_orientation_latched_ = false;
                return false;
            }

            // Tilted past the limit -- but require it to stay that way. See the note on
            // fall_check_hold_s_ in the header for why one sample is not enough on hardware.
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

            spdlog::warn(
                "State_Multitask: fall detected (tilt > limit {:.1f} deg, held {:.3f}s, {:.2f}s "
                "since last trigger)",
                bad_orientation_limit_ * 180.0f / static_cast<float>(M_PI),
                held_s,
                command_->trigger_step >= 0 ? command_->elapsed() : -1.0f);
            return true;
        },
        FSMStringMap.right.at("Passive"),
        "fall(bad_orientation)"
    );
}

void State_Multitask::enter()
{
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    command_->step_dt = env->step_dt;
    command_->reset();
    State_Flip::command = command_;  // hand the jump_command / jump_time terms their source
    bad_orientation_latched_ = false;

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
            // Before env->step(), so the jump_command / jump_time terms describe this step rather
            // than the previous one.
            command_->step();
            env->step();

            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_Multitask::run()
{
    if (keyboard && keyboard->on_pressed)
    {
        const std::string key = keyboard->key();
        for (const auto & m : motions_)
        {
            if (m.key == key)
            {
                // Queued, not applied: the command clock advances on the policy thread, and a
                // motion must start on a step boundary for jump_time to match training.
                command_->request(m.target_height, m.target_pitch_turns, m.target_roll_turns);
                break;
            }
        }
    }

    auto action = env->action_manager->processed_actions();
    for (int i(0); i < env->robot->data.joint_ids_map.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}

void State_Multitask::exit()
{
    policy_thread_running = false;
    if (policy_thread.joinable())
    {
        policy_thread.join();
    }
    // Leave State_Flip::command pointing at our (now idle) command rather than nulling it: the
    // observation terms already treat a disabled command as all-zero, and State_Flip overwrites the
    // pointer on its own enter().
}
