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
// `velocity_commands` (joystick); this state rewrites that one key to `biped_velocity_commands`
// (the keyboard drive, recorded for the log) at load time, in place so the ONNX input layout is
// unchanged.
//
// Logs (BipedLogger.h): 50 Hz telemetry, a 1 kHz torque capture over the rise, and a summary on
// exit, in `log_dir` (config.yaml; default deploy/robots/go2w/output).

#pragma once

#include "FSM/FSMState.h"
#include "JointActionMap.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class BipedLogger;

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

    // The velocity command the policy saw on its last step, written by biped_velocity_commands.
    static std::vector<float> last_command();
    static void record_command(const std::vector<float> & command);

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unique_ptr<JointActionMap> action_map_;
    float stance_ = 1.0f;

    // The stance itself is a 70-90 degree tilt, so the usual 1.0 rad limit would read success
    // as a fall; this one only catches the robot going over (Go2's feat/biped uses 2.6).
    float tilt_limit_ = 2.6f;

    std::unique_ptr<BipedLogger> logger_;

    std::thread policy_thread;
    bool policy_thread_running = false;

    static std::mutex command_mutex_;
    static std::vector<float> command_;
};

REGISTER_FSM(State_Biped)
