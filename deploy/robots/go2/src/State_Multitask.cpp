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

// Defaults for a motion that names itself and gives no explicit targets.
//
// The four rotations are two mirrored pairs, and the sign is the only thing that separates each
// pair: a handspring is a backflip's pitch reversed, a right sideflip is a sideflip's roll
// reversed. That is exactly how the training task defines them -- it negates the sampled target
// rather than carrying a second range -- so the names here mean what they mean there.
//
// `sideflip` is -1.0, unlike State_Flip's +1.0. State_Flip's value is right for
// Go2-Sideflip-Double; every multi-task policy trained on target_roll_turns_range=(-1.0, -1.0),
// and commanding +1.0 asks it to roll the way it never learned. That mismatch is what made the
// sideflip fail in MuJoCo while the backflip and jump worked.
void apply_targets(YAML::Node node, const std::string & motion, float & h, float & p, float & r)
{
    h = node["target_height"] ? node["target_height"].as<float>() : 0.0f;
    p = node["target_pitch_turns"] ? node["target_pitch_turns"].as<float>() : 0.0f;
    r = node["target_roll_turns"] ? node["target_roll_turns"].as<float>() : 0.0f;
    if (h == 0.0f && p == 0.0f && r == 0.0f)
    {
        if (motion == "jump")                h = 0.20f;   // vertical, no rotation
        else if (motion == "backflip")       p = -1.0f;   // backward: front hips lifted
        else if (motion == "handspring")     p = +1.0f;   // forward:  rear hips lifted
        else if (motion == "sideflip")       r = -1.0f;   // left:     right hips lifted
        else if (motion == "sideflip_right") r = +1.0f;   // right:    left hips lifted
    }
}
}  // namespace

std::shared_ptr<State_Multitask::HandstandCommand> State_Multitask::handstand = nullptr;

namespace isaaclab
{
namespace mdp
{

// handstand_command : [enabled, stance * enabled]
// Mirrors HandstandCommand.command (handstand.py). `stance` is +1 for the front-leg stance and -1
// for the hind-leg one, and is zeroed while disabled -- exactly as the training term does, so the
// idle observation is the all-zero vector every other Go2 policy also produces here.
REGISTER_OBSERVATION(handstand_command)
{
    (void)env;
    (void)params;
    auto & cmd = State_Multitask::handstand;

    std::vector<float> obs(2, 0.0f);
    if (cmd)
    {
        const float enabled = cmd->enabled ? 1.0f : 0.0f;
        obs[0] = enabled;
        obs[1] = cmd->stance * enabled;
    }
    return obs;
}

}  // namespace mdp
}  // namespace isaaclab

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
    handstand_ = std::make_shared<HandstandCommand>();
    if (cfg["handstand_hold_s"])      handstand_->hold_duration_s = cfg["handstand_hold_s"].as<float>();
    if (cfg["handstand_rearm_s"])     handstand_->rearm_delay_s = cfg["handstand_rearm_s"].as<float>();
    if (cfg["stance_settle_s"])       stance_settle_s_ = cfg["stance_settle_s"].as<float>();
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

    if (cfg["stances"] && cfg["stances"].IsSequence())
    {
        for (const auto & entry : cfg["stances"])
        {
            StancePreset preset;
            preset.key = entry["key"].as<std::string>();
            preset.name = to_lower(entry["stance"].as<std::string>());
            if (preset.name == "front")     preset.stance = 1.0f;
            else if (preset.name == "hind") preset.stance = -1.0f;
            else
            {
                spdlog::error(
                    "State_{}: stance '{}' is neither 'front' nor 'hind'; skipping key '{}'.",
                    state_string, preset.name, preset.key);
                continue;
            }
            stances_.push_back(preset);
            spdlog::info("State_{}: stance key '{}' -> {} (stance {:+.0f})",
                         state_string, preset.key, preset.name, preset.stance);
        }
    }
    else
    {
        spdlog::warn(
            "State_{}: no `stances:` list configured -- the bipedal stances cannot be commanded. "
            "The policy still runs; handstand_command simply stays zero.", state_string);
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

            // A bipedal stance puts the trunk through exactly the tilt this guard watches for --
            // that is what the stance *is*. Suppressed while the flag is high and through the
            // settle back onto four legs, which the policy learned as part of the episode rather
            // than as a fall. Without this the guard drops every successful handstand to Passive,
            // and it does so with the robot up on two legs, which is the worst moment to go limp.
            if (handstand_)
            {
                const float since_stance = handstand_->elapsed();
                if (handstand_->enabled
                    || (since_stance >= 0.0f
                        && since_stance < handstand_->hold_duration_s + stance_settle_s_))
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
    handstand_->reset();
    handstand_->step_dt = env->step_dt;
    State_Multitask::handstand = handstand_;  // and the handstand_command term its own
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
            handstand_->step();
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
        bool handled = false;
        for (const auto & m : motions_)
        {
            if (m.key == key)
            {
                // Refused rather than queued while a stance is up. The training environment keeps
                // the two commands mutually exclusive, so a flip fired from two legs is a state
                // the policy has never been in -- and it is a long way to fall from.
                if (handstand_->enabled)
                {
                    spdlog::warn(
                        "State_Multitask: '{}' ({}) ignored -- a bipedal stance is active. Release "
                        "it first (press its key again).", key, m.name);
                }
                else
                {
                    // Queued, not applied: the command clock advances on the policy thread, and a
                    // motion must start on a step boundary for jump_time to match training.
                    command_->request(m.target_height, m.target_pitch_turns, m.target_roll_turns);
                }
                handled = true;
                break;
            }
        }
        if (!handled)
        {
            for (const auto & st : stances_)
            {
                if (st.key != key) continue;
                if (handstand_->enabled)
                {
                    // Same key (or the other stance key) while up = come down now. The auto-release
                    // at hold_duration_s still applies; this only lets the operator end it early.
                    handstand_->cancel();
                    spdlog::info("State_Multitask: releasing the {} stance", st.name);
                }
                else if (command_->enabled)
                {
                    spdlog::warn(
                        "State_Multitask: '{}' ({} stance) ignored -- an acrobatic move is running.",
                        key, st.name);
                }
                else
                {
                    handstand_->request(st.stance);
                    spdlog::info("State_Multitask: {} stance requested", st.name);
                }
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
    handstand_->reset();
    // Nulled, unlike State_Flip::command below: no other state publishes a handstand command, so
    // leaving ours in place would have the term reading a stale struct nothing steps any more.
    State_Multitask::handstand = nullptr;
    // Leave State_Flip::command pointing at our (now idle) command rather than nulling it: the
    // observation terms already treat a disabled command as all-zero, and State_Flip overwrites the
    // pointer on its own enter().
}
