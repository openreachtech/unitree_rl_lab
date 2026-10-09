// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// State_Flip: deploy-side controller for the Go2W jump / backflip / sideflip policies
// trained with `unitree_rl_lab.tasks.dynamic` (JumpCommand, Go2w-Jump-Phase2). Ported from
// the Go2 controller on feat/jump (same keyboard workflow), with the Go2W action layout:
// leg position targets + wheel velocity targets (see JointActionMap.h).
//
// The policy consumes two observation terms the locomotion policies do not have:
//   * jump_command : [enabled, target_height, target_pitch_turns, target_roll_turns]
//   * jump_time    : cubic time encoding measured from the command rising edge
// Both are reproduced in State_Flip.cpp from the live FlipCommand owned by this state.
//
// Motions are keyboard presets (config.yaml `motions:`): each key fires one motion on
// demand, the command drops back to zero afterwards (the policy holds a stand), and the
// state re-arms after `rearm_delay_s` so motions can be chained without leaving it.

#pragma once

#include "FSM/FSMState.h"
#include "JointActionMap.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

class State_Flip : public FSMState
{
public:
    // Mirrors the timing / target bookkeeping of JumpCommand (tasks/dynamic/mdp/commands.py),
    // reduced to the single-robot deploy case and driven by policy steps.
    struct FlipCommand
    {
        // Active targets (the 4-D policy command), latched from pending_* on each trigger.
        float target_height = 0.0f;       // m above the nominal standing height
        float target_pitch_turns = 0.0f;  // backflip: -1.0 == one backward rotation
        float target_roll_turns = 0.0f;   // sideflip

        // Written by the FSM (keyboard) thread, read on trigger by the policy thread.
        std::atomic<float> pending_height{0.0f};
        std::atomic<float> pending_pitch{0.0f};
        std::atomic<float> pending_roll{0.0f};
        std::atomic<bool> trigger_requested{false};

        // Must match the training JumpCommandCfg.
        float time_scale = 1.0f;          // jump_time_encoding time scale
        float command_duration_s = 0.5f;  // how long `enabled` stays high after the trigger
        float rearm_delay_s = 0.3f;       // cooldown after a motion before the next can fire

        float step_dt = 0.02f;
        long step_count = 0;
        long trigger_step = -1;
        bool enabled = false;
        bool command_issued = false;

        void reset()
        {
            step_count = 0;
            trigger_step = -1;
            enabled = false;
            command_issued = false;
            trigger_requested.store(false);
            target_height = 0.0f;
            target_pitch_turns = 0.0f;
            target_roll_turns = 0.0f;
        }

        // Queue a motion. Called from the FSM thread.
        void request(float height, float pitch_turns, float roll_turns)
        {
            pending_height.store(height);
            pending_pitch.store(pitch_turns);
            pending_roll.store(roll_turns);
            trigger_requested.store(true);
        }

        // Advance one policy step; call once *before* the observation is computed.
        void step()
        {
            step_count += 1;

            // Once a motion has finished and the cooldown elapsed, re-arm. Targets drop to
            // zero in the gap so the policy holds a stand between motions.
            if (command_issued && !enabled && elapsed() >= command_duration_s + rearm_delay_s)
            {
                command_issued = false;
                target_height = 0.0f;
                target_pitch_turns = 0.0f;
                target_roll_turns = 0.0f;
            }

            if (!command_issued && trigger_requested.exchange(false))
            {
                target_height = pending_height.load();
                target_pitch_turns = pending_pitch.load();
                target_roll_turns = pending_roll.load();
                enabled = true;
                command_issued = true;
                trigger_step = step_count;
            }

            // The command expires after command_duration_s; afterwards the policy sees a
            // zero command and returns to standing.
            if (enabled && elapsed() >= command_duration_s)
            {
                enabled = false;
            }
        }

        float elapsed() const
        {
            return trigger_step >= 0 ? (step_count - trigger_step) * step_dt : 0.0f;
        }

        // Like jump_time_encoding, the encoded time is gated by `enabled`.
        float time_since_trigger() const { return enabled ? elapsed() : 0.0f; }
    };

    struct MotionPreset
    {
        std::string key;
        std::string name;
        float target_height = 0.0f;
        float target_pitch_turns = 0.0f;
        float target_roll_turns = 0.0f;
    };

    State_Flip(int state_mode, std::string state_string);

    void enter();
    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable())
        {
            policy_thread.join();
        }
    }

    // Read by the jump_command / jump_time observation terms. Set on enter().
    static std::shared_ptr<FlipCommand> command;

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unique_ptr<JointActionMap> action_map_;
    std::shared_ptr<FlipCommand> command_;
    std::vector<MotionPreset> motions_;

    // Fall -> Passive guard, armed only after the motion window plus this delay, so the
    // (expected) upside-down flight and the landing settle never trip it.
    float fall_check_delay_s_ = 1.5f;
    float bad_orientation_limit_ = 1.0f;
    // The tilt must exceed the limit continuously for this long: a hard touchdown swings
    // the IMU's fused gravity estimate for a few milliseconds.
    float fall_check_hold_s_ = 0.1f;
    std::chrono::steady_clock::time_point bad_orientation_since_{};
    bool bad_orientation_latched_ = false;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_Flip)
