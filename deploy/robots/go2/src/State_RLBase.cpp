#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

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

    // Drop to Passive once the trunk tilts past this many radians. A policy that is meant to get up
    // from a fall (GetUp, Recovery) sets `bad_orientation_limit: 0` in config.yaml to disable it.
    float bad_orientation_limit = cfg["bad_orientation_limit"] ? cfg["bad_orientation_limit"].as<float>() : 1.0f;
    if (bad_orientation_limit > 0.0f) {
        this->registered_checks.emplace_back(
            std::make_pair(
                [this, bad_orientation_limit]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), bad_orientation_limit); },
                FSMStringMap.right.at("Passive")
            )
        );
    }
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}