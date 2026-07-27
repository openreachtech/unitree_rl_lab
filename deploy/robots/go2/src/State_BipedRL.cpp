#include "State_BipedRL.h"

#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "KeyboardVelocityCommand.h"
#include "param.h"

#include <algorithm>
#include <map>
#include <utility>
#include <spdlog/spdlog.h>

namespace
{

// Margin below tilt_limit_ the base has to reach before the strict quadruped limit is
// armed, so that arming right on the boundary cannot trip on sensor noise alone [rad].
constexpr float kQuadTiltArmMargin = 0.1f;

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
{
    auto cfg = param::config["FSM"][state_string];

    if (cfg["biped_tilt_limit"].IsDefined())
    {
        biped_tilt_limit_ = cfg["biped_tilt_limit"].as<float>();
    }

    load_mode_triggers(cfg, state_string);

    if (mode_triggers_.empty())
    {
        spdlog::warn(
            "FSM: State_{} has no gait_modes / keyboard_gait_modes bindings; "
            "the policy will stay in {} mode.",
            state_string,
            go2::gait_mode_name(go2::GaitMode::kQuad));
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
    // GaitModeCommand forces the first mode of every training episode to quad and only
    // switches once the policy is already running, so entry mirrors that.
    go2::GaitModeSelector::instance().set(go2::GaitMode::kQuad);
    spdlog::info("Gait mode: {}", go2::gait_mode_name(go2::GaitMode::kQuad));
    quad_tilt_limit_armed_ = false;

    State_RLBase::enter();
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
