#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

namespace isaaclab
{
// deploy.yaml: use observation key "keyboard_velocity_commands" (not "velocity_commands").
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    if (!FSMState::keyboard)
    {
        FSMState::keyboard = std::make_shared<Keyboard>();
    }

    std::string key = FSMState::keyboard->key();
    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key == "w")
    {
        cmd = {cfg["lin_vel_x"][1].as<float>(), 0.0f, 0.0f};
    }
    else if (key == "s")
    {
        cmd = {cfg["lin_vel_x"][0].as<float>(), 0.0f, 0.0f};
    }
    else if (key == "a")
    {
        cmd = {0.0f, cfg["lin_vel_y"][1].as<float>(), 0.0f};
    }
    else if (key == "d")
    {
        cmd = {0.0f, cfg["lin_vel_y"][0].as<float>(), 0.0f};
    }
    else if (key == "q")
    {
        cmd = {0.0f, 0.0f, cfg["ang_vel_z"][1].as<float>()};
    }
    else if (key == "e")
    {
        cmd = {0.0f, 0.0f, cfg["ang_vel_z"][0].as<float>()};
    }
    return cmd;
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