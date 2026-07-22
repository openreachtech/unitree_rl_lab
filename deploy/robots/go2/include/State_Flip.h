// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// State_Flip: deploy-side controller for the Go2 dynamic jump / backflip /
// sideflip policies trained with `unitree_rl_lab.tasks.dynamic` (JumpCommand).
//
// The trained policy consumes two extra observation terms compared to the
// locomotion policy:
//   * jump_command : [enabled, target_height, target_pitch_turns, target_roll_turns]
//   * jump_time    : cubic time encoding measured from the command rising edge
// Both are reproduced here (see State_Flip.cpp) from the live `FlipCommand`
// state owned by the active flip state.
//
// One state class powers every motion; the concrete motion (jump / backflip /
// sideflip) is selected purely by the command targets read from config.yaml.

#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

class State_Flip : public FSMState
{
public:
    // Mirrors the timing / target bookkeeping of `JumpCommand` in
    // source/.../tasks/dynamic/mdp/commands.py, reduced to the single-robot
    // deploy case. All time-keeping is self-contained (driven by policy steps)
    // so the observation terms never depend on env->episode_length ordering.
    struct FlipCommand
    {
        // --- active motion targets (mirror the 4-D policy command vector) ---
        // Latched from the pending_* values on each trigger; read by the
        // jump_command observation term.
        float target_height = 0.0f;        // meters above nominal standing height
        float target_pitch_turns = 0.0f;   // backflip target, in turns (-1.0 == one back rotation)
        float target_roll_turns = 0.0f;    // sideflip target, in turns

        // --- pending targets (written by the keyboard/FSM thread, read on trigger) ---
        std::atomic<float> pending_height{0.0f};
        std::atomic<float> pending_pitch{0.0f};
        std::atomic<float> pending_roll{0.0f};
        std::atomic<bool> trigger_requested{false};

        // --- timing (must match the training JumpCommandCfg) ---
        float time_scale = 1.0f;           // jump_time_encoding time scale
        float command_duration_s = 0.5f;   // how long `enabled` stays high after trigger
        float trigger_delay_s = 1.0f;      // (single-shot mode) auto-trigger this long after entry
        float rearm_delay_s = 0.3f;        // (manual mode) cooldown after a motion before re-firing
        bool use_auto_trigger = true;      // single-shot: fire once automatically after trigger_delay_s
        bool manual_mode = false;          // manual: fire on demand + re-arm, target chosen per trigger

        // --- runtime state ---
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

        // Queue a motion (manual mode). Called from the FSM thread.
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

            // Manual mode: once a motion has finished and the cooldown elapsed,
            // re-arm so the next keypress can fire again. Targets drop to zero in
            // the gap so the policy holds a stand between motions.
            if (manual_mode && command_issued && !enabled
                && elapsed() >= command_duration_s + rearm_delay_s)
            {
                command_issued = false;
                target_height = 0.0f;
                target_pitch_turns = 0.0f;
                target_roll_turns = 0.0f;
            }

            if (!command_issued)
            {
                bool fire = trigger_requested.exchange(false);
                if (use_auto_trigger && (step_count * step_dt) >= trigger_delay_s)
                {
                    fire = true;
                }
                if (fire)
                {
                    if (manual_mode)
                    {
                        target_height = pending_height.load();
                        target_pitch_turns = pending_pitch.load();
                        target_roll_turns = pending_roll.load();
                    }
                    enabled = true;
                    command_issued = true;
                    trigger_step = step_count;
                }
            }

            // The command expires after command_duration_s; afterwards the
            // policy sees a zero command and returns to standing.
            if (enabled && elapsed() >= command_duration_s)
            {
                enabled = false;
            }
        }

        float elapsed() const
        {
            return trigger_step >= 0 ? (step_count - trigger_step) * step_dt : 0.0f;
        }

        // Retained after the command turns off? No -- like jump_time_encoding,
        // the encoded time is gated by `enabled` so it drops back to zero.
        float time_since_trigger() const { return enabled ? elapsed() : 0.0f; }
    };

    // A keyboard-selectable motion preset (manual/Dynamic mode).
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

    // Shared with the observation terms (jump_command / jump_time) of the
    // currently active flip state. Set on enter(), like State_Mimic::motion.
    static std::shared_ptr<FlipCommand> command;

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<FlipCommand> command_;

    std::string trigger_key_;             // (single-shot) optional key to fire the one configured motion
    std::vector<MotionPreset> motions_;   // (manual/Dynamic) key -> motion presets
    bool manual_mode_ = false;            // true when a `motions:` list is configured
    float fall_check_delay_s_ = 0.3f;     // grace period after the motion before fall -> Passive
    float bad_orientation_limit_ = 1.2f;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_Flip)
