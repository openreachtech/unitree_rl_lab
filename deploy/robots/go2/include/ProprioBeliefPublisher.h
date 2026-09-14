#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <array>
#include <mutex>

namespace go2
{

// Publishes the 45-dim proprioception vector belief_encoder_node.py's frozen
// BeliefTerrainEncoder needs alongside the raw/noisy heightmap, on
// /go2_ctrl/proprio_belief (std_msgs/Float32MultiArray). See that file's module
// docstring for the exact contract this must match:
//
//   [0:3]   base_ang_vel        = env->robot->data.root_ang_vel_b        * 0.2
//   [3:6]   projected_gravity   = env->robot->data.projected_gravity_b   * 0.05
//   [6:9]   base_velocity cmd   = latest keyboard_velocity_commands cmd  (unscaled)
//   [9:21]  joint_pos_rel       = (joint_pos - default_joint_pos)        * 0.01
//   [21:33] joint_vel_rel       = joint_vel                              * 0.05
//   [33:45] last_action         = env->action_manager->action()         (unscaled)
//
// These scales are the *encoder's* training-time scales (belief_height_map.py's
// GO2_BLIND_PROPRIO_SPEC), not necessarily the actor's own observation scales in
// deploy.yaml - they happen to agree for base_ang_vel/joint_vel_rel/last_action/the
// velocity command in the currently deployed checkpoints, but NOT for
// projected_gravity (actor scale 1.0) or joint_pos_rel (actor scale 1.0), so this
// reads the raw env data directly rather than reusing the actor's own obs terms.
//
// Harmless to always run (init() + publish() every policy step) even when no
// belief_encoder_node is subscribed - one extra small topic nobody reads.
class ProprioBeliefPublisher
{
public:
    static ProprioBeliefPublisher& instance();

    void init();

    // Called from State_RLBase.cpp's keyboard_velocity_commands observation each
    // time it recomputes the latched command, so publish() always has the most
    // recent one available (env->cfg's command ranges aren't otherwise reachable
    // from here without re-parsing YAML).
    void set_velocity_command(const std::array<float, 3>& cmd);

    // Called once per policy step (State_RLBase.h's policy_thread, right after
    // env->step()) so the published proprioception matches what the actor itself
    // just observed this same step. No-op if init() was never called.
    void publish(isaaclab::ManagerBasedRLEnv* env);

private:
    ProprioBeliefPublisher() = default;

    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;

    std::mutex cmd_mutex_;
    std::array<float, 3> velocity_command_{0.0f, 0.0f, 0.0f};
};

} // namespace go2
