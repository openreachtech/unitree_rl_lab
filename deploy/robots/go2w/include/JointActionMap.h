#pragma once

#include "Types.h"

#include <yaml-cpp/yaml.h>

#include <vector>

// Maps a Go2W policy's processed actions onto SDK motors. Shared by every RL state.
//
// processed_actions() concatenates the action terms in deploy.yaml order (leg positions,
// then wheel velocities), and element k of a term refers to that term's joint_ids[k] --
// an *IsaacLab* joint id. joint_ids_map is itself indexed by IsaacLab joint id, so the
// SDK motor is joint_ids_map[joint_ids[k]]; indexing it with the action index k directly
// is only correct when a term's joint_ids happen to be the identity.
//
// That identity holds on Go2, whose sole action term is joint_names=[".*"]. It does *not*
// hold on Go2W: the legs and wheels need separate position/velocity terms, so both declare
// explicit SDK-ordered joint_names with preserve_order=True and their joint_ids come out
// as permutations ([1,5,9,0,...] and [13,12,15,14]). Skipping the composition cross-wired
// 14 of the 16 joints -- FR_thigh's target drove FR_hip, FR_calf's drove RL_hip, and the
// left/right wheels were swapped -- which made the robot thrash the instant the policy
// took over.
class JointActionMap
{
public:
    explicit JointActionMap(const YAML::Node & deploy_cfg)
    {
        const auto joint_ids_map = deploy_cfg["joint_ids_map"].as<std::vector<int>>();
        pos_motor_ids_ = resolve(deploy_cfg["actions"]["JointPositionAction"], joint_ids_map);
        vel_motor_ids_ = resolve(deploy_cfg["actions"]["JointVelocityAction"], joint_ids_map);
    }

    // Leg targets go to q, wheel targets to dq (the wheels are velocity-controlled: kp 0).
    void write(const std::vector<float> & action, LowCmd_t & lowcmd) const
    {
        for (size_t i = 0; i < pos_motor_ids_.size(); ++i)
        {
            lowcmd.msg_.motor_cmd()[pos_motor_ids_[i]].q() = action[i];
        }
        const size_t vel_offset = pos_motor_ids_.size();
        for (size_t i = 0; i < vel_motor_ids_.size(); ++i)
        {
            lowcmd.msg_.motor_cmd()[vel_motor_ids_[i]].dq() = action[vel_offset + i];
        }
    }

private:
    static std::vector<int> resolve(const YAML::Node & action_cfg, const std::vector<int> & joint_ids_map)
    {
        auto joint_ids_node = action_cfg["joint_ids"];
        if (!joint_ids_node || joint_ids_node.IsNull())
        {
            // No explicit selection: the term spans every joint in IsaacLab order, so the
            // action index *is* the IsaacLab joint id.
            return joint_ids_map;
        }
        std::vector<int> motor_ids;
        for (int isaac_joint_id : joint_ids_node.as<std::vector<int>>())
        {
            motor_ids.push_back(joint_ids_map.at(isaac_joint_id));
        }
        return motor_ids;
    }

    std::vector<int> pos_motor_ids_;
    std::vector<int> vel_motor_ids_;
};
