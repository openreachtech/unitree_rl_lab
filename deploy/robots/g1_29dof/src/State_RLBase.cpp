#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "param.h"
#include <algorithm>
#include <array>

namespace isaaclab
{
// Keyboard velocity commands (same keys as go2 on feat/mid360).
// deploy.yaml must name the observation "keyboard_velocity_commands"
// (train.py --deploy-keyboard-commands does this).
//
//   f / b : forward / backward     l / r : left / right     y / u : turn left / right
//   keys latch (press once, combine freely); space zeroes the command.
//
// Each latched key requests keyboard_vel_scale x the trained limit in that direction,
// low-pass filtered by keyboard_alpha. Both are read from config.yaml
// (FSM.Velocity.keyboard_vel_scale / keyboard_alpha), falling back to
// deploy.yaml commands.base_velocity, then to the defaults below.
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    auto keyboard = FSMState::keyboard;
    const auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];
    const auto deploy_cmd = env->cfg["commands"]["base_velocity"];
    const auto fsm_cfg = param::config["FSM"]["Velocity"];

    auto read = [&](const char* key, float fallback) {
        if (fsm_cfg[key].IsDefined()) return fsm_cfg[key].as<float>();
        if (deploy_cmd[key].IsDefined()) return deploy_cmd[key].as<float>();
        return fallback;
    };
    const float vel_scale = read("keyboard_vel_scale", 0.5f);
    const float alpha = read("keyboard_alpha", 0.05f);

    const auto lim = [&](const char* axis, int idx) { return vel_scale * ranges[axis][idx].as<float>(); };

    static std::array<float, 3> cmd = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> target = {0.0f, 0.0f, 0.0f};

    if (keyboard->consume_velocity_stop())
    {
        cmd = {0.0f, 0.0f, 0.0f};
        return std::vector<float>(cmd.begin(), cmd.end());
    }

    if (keyboard->pressed("f")) target[0] += lim("lin_vel_x", 1);
    if (keyboard->pressed("b")) target[0] += lim("lin_vel_x", 0);
    if (keyboard->pressed("l")) target[1] += lim("lin_vel_y", 1);
    if (keyboard->pressed("r")) target[1] += lim("lin_vel_y", 0);
    if (keyboard->pressed("y")) target[2] += lim("ang_vel_z", 1);
    if (keyboard->pressed("u")) target[2] += lim("ang_vel_z", 0);

    for (int i = 0; i < 3; ++i)
    {
        cmd[i] = (1.0f - alpha) * cmd[i] + alpha * target[i];
    }
    cmd[0] = std::clamp(cmd[0], lim("lin_vel_x", 0), lim("lin_vel_x", 1));
    cmd[1] = std::clamp(cmd[1], lim("lin_vel_y", 0), lim("lin_vel_y", 1));
    cmd[2] = std::clamp(cmd[2], lim("ang_vel_z", 0), lim("ang_vel_z", 1));
    return std::vector<float>(cmd.begin(), cmd.end());
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