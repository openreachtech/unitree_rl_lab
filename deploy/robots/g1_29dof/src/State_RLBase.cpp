#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "param.h"
#include "CsvRecorder.h"
#include <algorithm>
#include <array>

namespace isaaclab
{
// Velocity commands from the keyboard and the joystick, both scaled to the trained limits.
// deploy.yaml must name the observation "keyboard_velocity_commands"
// (train.py --deploy-keyboard-commands does this).
//
// Keyboard (g1_ctrl terminal):
//   f / b : forward / backward     l / r : left / right     y / u : turn left / right
//   keys latch (press once, combine freely); space zeroes the command.
//   A latched key asks for keyboard_vel_scale x the trained limit in that direction,
//   low-pass filtered by keyboard_alpha (the keys are steps).
// Joystick: left stick vx / vy, right stick x yaw rate. Full deflection is
//   joystick_vel_scale x the trained limit in that direction (forward and backward
//   limits differ), proportional in between.
//
// The two add up (in practice one is used at a time). Scales and alpha come from
// config.yaml FSM.Velocity.
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    const auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];
    const auto cfg = param::config["FSM"]["Velocity"];
    const float kb_scale = cfg["keyboard_vel_scale"].as<float>(0.5f);
    const float joy_scale = cfg["joystick_vel_scale"].as<float>(0.8f);
    const float alpha = cfg["keyboard_alpha"].as<float>(0.05f);

    const char* axes[3] = {"lin_vel_x", "lin_vel_y", "ang_vel_z"};
    float lo[3], hi[3];
    for (int i = 0; i < 3; ++i)
    {
        lo[i] = ranges[axes[i]][0].as<float>();
        hi[i] = ranges[axes[i]][1].as<float>();
    }

    // keyboard: latched direction (-1 / 0 / +1 per axis), low-pass filtered
    auto keyboard = FSMState::keyboard;
    static std::array<float, 3> kb = {0.0f, 0.0f, 0.0f};
    if (keyboard->consume_velocity_stop()) kb = {0.0f, 0.0f, 0.0f};
    const float dir[3] = {
        float(keyboard->pressed("f")) - float(keyboard->pressed("b")),
        float(keyboard->pressed("l")) - float(keyboard->pressed("r")),
        float(keyboard->pressed("y")) - float(keyboard->pressed("u")),
    };

    // joystick: stick in [-1, 1] (x forward / left / counter-clockwise positive)
    auto& joy = env->robot->data.joystick;
    const float stick[3] = {joy->ly(), -joy->lx(), -joy->rx()};

    std::vector<float> cmd(3);
    for (int i = 0; i < 3; ++i)
    {
        const float kb_target = kb_scale * (dir[i] > 0 ? hi[i] : -lo[i]) * dir[i];
        kb[i] = (1.0f - alpha) * kb[i] + alpha * kb_target;
        const float s = std::clamp(stick[i], -1.0f, 1.0f);
        const float joy_cmd = joy_scale * (s > 0 ? hi[i] : -lo[i]) * s;
        cmd[i] = std::clamp(kb[i] + joy_cmd, lo[i], hi[i]);
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    if (auto log_dir = cfg["log_dir"].as<std::string>(""); !log_dir.empty())
    {
        std::filesystem::path dir(log_dir);
        recorder = std::make_unique<CsvRecorder>(dir.is_relative() ? param::proj_dir / dir : dir);
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}