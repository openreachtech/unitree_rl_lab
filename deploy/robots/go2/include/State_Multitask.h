// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

// State_Multitask -- runs the merged locomotion + acrobatics policy.
//
// The multi-task network reads one 122-column observation that contains BOTH task selectors:
//
//   velocity_commands / keyboard_velocity_commands : 3   (driving)
//   jump_command                                   : 4   (which acrobatic move, if any)
//   jump_time                                      : 1   (phase within that move)
//
// Every one of those observation terms already exists on this side -- the velocity ones in
// observations.h and State_RLBase.cpp, the jump ones in State_Flip.cpp. What did not exist is a
// state that produces both at once: State_RLBase leaves State_Flip::command null, so the policy
// reads a permanently-zero jump command, and State_Flip has no velocity source. This state is the
// missing combination, and deliberately adds no new observation code.
//
// It reuses State_Flip::FlipCommand rather than reimplementing the command clock. That struct
// encodes the timing the policy was trained against -- how long `enabled` stays high, the cubic
// jump_time encoding, the re-arm cooldown -- and has already been validated against MuJoCo for the
// acrobatics policy. A second implementation would be a second thing to keep in sync.
//
// Difference from State_Flip that matters most: after a move, this policy is expected to go back to
// *running*, not to stand. The fall guard below therefore has to distinguish "landed and carried on"
// from "fell over", which is exactly what State_Flip's tuned delay-and-hold does -- so those values
// are inherited rather than re-derived.

#include "FSM/FSMState.h"
#include "State_Flip.h"

#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/terminations.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

class State_Multitask : public FSMState
{
public:
    State_Multitask(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit();

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    // Shared with the jump_command / jump_time observation terms through
    // State_Flip::command, the same channel State_Flip itself publishes on.
    std::shared_ptr<State_Flip::FlipCommand> command_;
    std::vector<State_Flip::MotionPreset> motions_;

    // --- fall guard ---------------------------------------------------------
    // Values inherited from State_Flip, where they were tuned on hardware (see feat/jump commit
    // e4218d9, "update controller to solve backflip policy exiting issue"). Both matter for a
    // different reason:
    //
    //   fall_check_delay_s_  the guard must stay off until well past touchdown. At 0.3 s it armed
    //                        at the instant the front feet slammed down with the trunk still
    //                        pitching, and dropped good backflips to Passive.
    //   fall_check_hold_s_   the tilt has to exceed the limit *continuously*. The real IMU's fused
    //                        orientation swings for a few milliseconds under landing shock, and
    //                        this check runs at the 1 kHz FSM rate, so one glitched sample would
    //                        otherwise end the run. Simulation never reproduces this -- its
    //                        orientation is ground truth -- so it cannot be tuned away in MuJoCo.
    float fall_check_delay_s_ = 1.2f;
    float fall_check_hold_s_ = 0.1f;
    float bad_orientation_limit_ = 1.2f;
    std::chrono::steady_clock::time_point bad_orientation_since_{};
    bool bad_orientation_latched_ = false;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_Multitask)
