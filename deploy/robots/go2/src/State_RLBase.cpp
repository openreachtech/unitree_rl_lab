#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <array>
#include <algorithm>

namespace isaaclab
{

// deploy.yaml:
//   observations: keyboard_velocity_commands  (not velocity_commands)
//   commands.base_velocity:
//     keyboard_vel_scale: 0.5   # optional, default 0.5
//     keyboard_alpha: 0.15      # optional low-pass; higher = smoother / slower

REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    if (!FSMState::keyboard)
    {
        FSMState::keyboard = std::make_shared<Keyboard>();
    }

    auto keyboard = FSMState::keyboard;
    const auto cmd_cfg = env->cfg["commands"]["base_velocity"];
    const auto ranges = cmd_cfg["ranges"];

    float vel_scale = 0.5f;
    if (cmd_cfg["keyboard_vel_scale"].IsDefined())
    {
        vel_scale = cmd_cfg["keyboard_vel_scale"].as<float>();
    }

    float alpha = 0.15f;
    if (cmd_cfg["keyboard_alpha"].IsDefined())
    {
        alpha = cmd_cfg["keyboard_alpha"].as<float>();
    }

    const auto sx = [&](int idx) { return vel_scale * ranges["lin_vel_x"][idx].as<float>(); };
    const auto sy = [&](int idx) { return vel_scale * ranges["lin_vel_y"][idx].as<float>(); };
    const auto sz = [&](int idx) { return vel_scale * ranges["ang_vel_z"][idx].as<float>(); };

    static std::array<float, 3> cmd = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> target = {0.0f, 0.0f, 0.0f};

    if (keyboard->consume_velocity_stop())
    {
        cmd = {0.0f, 0.0f, 0.0f};
        return std::vector<float>(cmd.begin(), cmd.end());
    }

    if (keyboard->pressed("f"))
    {
        target[0] += sx(1);
    }
    if (keyboard->pressed("b"))
    {
        target[0] += sx(0);
    }
    if (keyboard->pressed("l"))
    {
        target[1] += sy(1);
    }
    if (keyboard->pressed("r"))
    {
        target[1] += sy(0);
    }
    if (keyboard->pressed("y"))
    {
        target[2] += sz(1);
    }
    if (keyboard->pressed("u"))
    {
        target[2] += sz(0);
    }

    for (int i = 0; i < 3; ++i)
    {
        cmd[i] = (1.0f - alpha) * cmd[i] + alpha * target[i];
    }

    cmd[0] = std::clamp(cmd[0], sx(0), sx(1));
    cmd[1] = std::clamp(cmd[1], sy(0), sy(1));
    cmd[2] = std::clamp(cmd[2], sz(0), sz(1));

    return std::vector<float>(cmd.begin(), cmd.end());
}

} // namespace isaaclab

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
