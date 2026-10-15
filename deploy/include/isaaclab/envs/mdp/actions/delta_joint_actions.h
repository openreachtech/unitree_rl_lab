// Delta joint-position action of the G1 get-up task (tasks/locomotion/robots/g1_getup/mdp.py,
// DeltaJointPositionAction): the policy outputs a change relative to the *current* joint
// position, target = q_now + scale * a. processed_actions() returns the delta (scale * a); the
// FSM state adds the current joint position at every control tick (State_GetUp::run), as the
// training rebuilt the target at every physics step.

#pragma once

#include "isaaclab/envs/mdp/actions/joint_actions.h"

namespace isaaclab
{

class JointPositionDeltaAction : public JointAction
{
public:
    JointPositionDeltaAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    : JointAction(without_offset(cfg), env)
    {
        _action_clip = cfg["action_clip"] ? cfg["action_clip"].as<float>() : 100.0f;
        _unactuated_steps = cfg["unactuated_steps"] ? cfg["unactuated_steps"].as<int>() : 0;
    }

    void process_actions(std::vector<float> actions) override
    {
        for (auto & a : actions) a = std::clamp(a, -_action_clip, _action_clip);
        _raw_actions = actions;
        for (int i(0); i < _action_dim; ++i) {
            _processed_actions[i] = _scale.empty() ? _raw_actions[i] : _raw_actions[i] * _scale[i];
        }
    }

    float scale() const { return _scale.empty() ? 1.0f : _scale[0]; }
    int unactuated_steps() const { return _unactuated_steps; }

private:
    // deploy.yaml writes the training's `offset` (a scalar 0 here); a delta action does not use it, and
    // JointAction would read it as a per-joint list
    static YAML::Node without_offset(const YAML::Node & cfg)
    {
        YAML::Node c = YAML::Clone(cfg);
        c["offset"] = YAML::Node(YAML::NodeType::Null);  // JointAction skips a null offset
        return c;
    }

    float _action_clip;
    int _unactuated_steps;
};

REGISTER_ACTION(JointPositionDeltaAction);

};
