#include "ProprioBeliefPublisher.h"

#include <spdlog/spdlog.h>

namespace go2
{

ProprioBeliefPublisher& ProprioBeliefPublisher::instance()
{
    static ProprioBeliefPublisher publisher;
    return publisher;
}

void ProprioBeliefPublisher::init()
{
    node_ = rclcpp::Node::make_shared("go2_proprio_belief_publisher");
    pub_ = node_->create_publisher<std_msgs::msg::Float32MultiArray>(
        "/go2_ctrl/proprio_belief", rclcpp::QoS(10));
    spdlog::info(
        "ProprioBeliefPublisher: publishing 45-dim proprioception on /go2_ctrl/proprio_belief");
}

void ProprioBeliefPublisher::set_velocity_command(const std::array<float, 3>& cmd)
{
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    velocity_command_ = cmd;
}

void ProprioBeliefPublisher::publish(isaaclab::ManagerBasedRLEnv* env)
{
    if (!pub_)
    {
        return;
    }

    auto& data = env->robot->data;

    std_msgs::msg::Float32MultiArray msg;
    msg.data.resize(45);

    for (int i = 0; i < 3; ++i)
    {
        msg.data[i] = data.root_ang_vel_b[i] * 0.2f;
    }
    for (int i = 0; i < 3; ++i)
    {
        msg.data[3 + i] = data.projected_gravity_b[i] * 0.05f;
    }
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        for (int i = 0; i < 3; ++i)
        {
            msg.data[6 + i] = velocity_command_[i];
        }
    }
    for (int i = 0; i < 12; ++i)
    {
        msg.data[9 + i] = (data.joint_pos[i] - data.default_joint_pos[i]) * 0.01f;
    }
    for (int i = 0; i < 12; ++i)
    {
        msg.data[21 + i] = data.joint_vel[i] * 0.05f;
    }
    auto action = env->action_manager->action();
    for (int i = 0; i < 12; ++i)
    {
        msg.data[33 + i] = action[i];
    }

    pub_->publish(msg);
}

} // namespace go2
