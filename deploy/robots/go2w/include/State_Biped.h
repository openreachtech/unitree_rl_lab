// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// State_Biped: deploy-side controller for the Go2W handstand policies trained with
// `unitree_rl_lab.tasks.biped` (Go2w-Biped and its single-stance siblings). Keyboard workflow
// follows the Go2 controller on feat/biped: enter a stance from FixStand, drive it with the
// latched velocity keys, [0] drops to Passive.
//
// One state per stance, each with a fixed `stance` (+1 front wheels, -1 hind wheels) in
// config.yaml. The two-stance policy reads it from the `handstand_command` observation,
// [enabled, stance], reproduced here. A single-stance policy has no such term and simply
// ignores the setting.
//
// The policies observe the commanded velocity. Their deploy.yaml names the stock
// `velocity_commands` (joystick); this state rewrites that one key to
// `keyboard_velocity_commands` at load time, in place so the ONNX input layout is unchanged.

#pragma once

#include "FSM/FSMState.h"
#include "JointActionMap.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

class State_Biped : public FSMState
{
public:
    State_Biped(int state_mode, std::string state_string);
    ~State_Biped();

    void enter();
    void run();
    void exit();

    // Read by the handstand_command observation. Set on enter(): only one state runs at a time.
    static std::atomic<float> active_stance;

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unique_ptr<JointActionMap> action_map_;
    float stance_ = 1.0f;

    // The stance itself is a 70-90 degree tilt, so the usual 1.0 rad limit would read success
    // as a fall; this one only catches the robot going over (Go2's feat/biped uses 2.6).
    float tilt_limit_ = 2.6f;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_Biped)
