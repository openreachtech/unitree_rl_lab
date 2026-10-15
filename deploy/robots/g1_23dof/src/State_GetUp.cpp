#include "FSM/State_GetUp.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/observations/getup_observations.h"

State_GetUp::State_GetUp(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    auto action_cfg = deploy_cfg["actions"]["JointPositionDeltaAction"];
    if (!action_cfg) {
        throw std::runtime_error("GetUp: " + (policy_dir / "params" / "deploy.yaml").string()
                                 + " has no JointPositionDeltaAction (not a get-up policy?)");
    }
    unactuated_steps_ = action_cfg["unactuated_steps"] ? action_cfg["unactuated_steps"].as<int>() : 0;

    num_motors_ = cfg["num_motors"] ? cfg["num_motors"].as<int>() : 29;
    if (cfg["hold_kp"]) hold_kp_ = cfg["hold_kp"].as<float>();
    if (cfg["hold_kd"]) hold_kd_ = cfg["hold_kd"].as<float>();
    for (auto m : env->robot->data.joint_ids_map) policy_motors_.insert((int)m);

    spdlog::info("GetUp: policy {} ({} joints, unactuated {} steps)", policy_dir.string(),
                 env->robot->data.joint_ids_map.size(), unactuated_steps_);
}
