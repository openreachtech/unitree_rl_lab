// Observation terms of the G1 get-up task that the generic observations.h does not have under
// the names the get-up's deploy.yaml uses.

#pragma once

#include "isaaclab/envs/mdp/observations/observations.h"

namespace isaaclab
{
namespace mdp
{

// get-up task: `joint_vel` is Isaac Lab's joint_vel_rel (the default joint velocity is zero)
REGISTER_OBSERVATION(joint_vel)
{
    auto & asset = env->robot;
    std::vector<float> data(asset->data.joint_vel.data(), asset->data.joint_vel.data() + asset->data.joint_vel.size());
    return data;
}

// get-up task: the per-env action scale the policy sees; at deploy it is the final (deploy) scale
REGISTER_OBSERVATION(action_scale)
{
    auto actions = env->cfg["actions"]["JointPositionDeltaAction"];
    float scale = actions ? actions["scale"][0].as<float>() : 1.0f;
    return {scale};
}

}
}
