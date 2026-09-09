// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

// State_Multitask -- runs the merged locomotion + acrobatics policy.
//
// The multi-task network reads one 124-column observation carrying all three task selectors:
//
//   velocity_commands / keyboard_velocity_commands : 3   (driving)
//   jump_command                                   : 4   (which acrobatic move, if any)
//   jump_time                                      : 1   (phase within that move)
//   handstand_command                              : 2   (which bipedal stance, if any)
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
    // Mirrors HandstandCommand in source/.../tasks/biped/mdp/handstand.py, reduced to the
    // single-robot deploy case.
    //
    // Deliberately NOT a FlipCommand. A flip is a motion that runs to completion on its own clock,
    // so that struct is built around "fire, then measure elapsed time against the move"; a stance
    // is a mode that is entered and *held*, for 10 s in this policy against the flip's 1.0 s. The
    // observation shape differs for the same reason: there is no time encoding, only the flag and
    // which end is standing.
    struct HandstandCommand
    {
        // +1 front-leg stance, -1 hind-leg stance, 0 idle. The sign is the whole difference
        // between the two stances -- the policy reads it as one column.
        float stance = 0.0f;

        // Written by the FSM thread, read by the policy thread on the next step boundary. Same
        // reason FlipCommand queues rather than applies: the command clock advances on the policy
        // thread, and a stance has to begin on a step boundary to match training.
        std::atomic<float> pending_stance{0.0f};
        std::atomic<bool> trigger_requested{false};
        std::atomic<bool> cancel_requested{false};

        // Must match BIPED_HOLD_S in multitask_env_cfg_moe_v2.py. The policy has only ever seen the
        // flag held for this long and then released; it learned the descent as part of the episode.
        float hold_duration_s = 10.0f;
        float rearm_delay_s = 0.5f;

        float step_dt = 0.02f;
        long step_count = 0;
        long trigger_step = -1;
        bool enabled = false;

        void reset()
        {
            step_count = 0;
            trigger_step = -1;
            enabled = false;
            stance = 0.0f;
            trigger_requested.store(false);
            cancel_requested.store(false);
            pending_stance.store(0.0f);
        }

        // Seconds since the stance was last commanded; -1 before the first one.
        float elapsed() const
        {
            return trigger_step < 0 ? -1.0f : static_cast<float>(step_count - trigger_step) * step_dt;
        }

        void request(float stance_sign)
        {
            pending_stance.store(stance_sign);
            trigger_requested.store(true);
        }

        void cancel() { cancel_requested.store(true); }

        // Advance one policy step; call once *before* the observation is computed.
        void step()
        {
            step_count += 1;

            if (enabled && cancel_requested.exchange(false))
            {
                enabled = false;
                stance = 0.0f;
            }
            else if (enabled && elapsed() >= hold_duration_s)
            {
                // Auto-release at the trained hold length. Left high indefinitely the policy would
                // be in a state it never saw: every stance it learned ended by the flag dropping.
                enabled = false;
                stance = 0.0f;
            }

            if (trigger_requested.load() && !enabled)
            {
                const bool cooled = trigger_step < 0 || elapsed() >= hold_duration_s + rearm_delay_s;
                if (cooled)
                {
                    trigger_requested.store(false);
                    cancel_requested.store(false);
                    stance = pending_stance.load();
                    enabled = true;
                    trigger_step = step_count;
                }
            }
        }
    };

    // A keyboard-selectable stance, read from the `stances:` list in config.yaml.
    struct StancePreset
    {
        std::string key;
        std::string name;    // "front" / "hind", for logging
        float stance = 0.0f;
    };

    // Shared with the handstand_command observation term, the same channel State_Flip::command
    // uses for the jump terms. Null outside this state, and the term reads zeros then -- which is
    // what every other policy expects, since none of them has this column.
    static std::shared_ptr<HandstandCommand> handstand;

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

    std::shared_ptr<HandstandCommand> handstand_;
    std::vector<StancePreset> stances_;

    // Extra seconds the fall guard stays suppressed after a stance releases. The robot comes down
    // from two legs through the same tilt range the guard is watching for, and the descent is part
    // of what the policy learned rather than a fall.
    float stance_settle_s_ = 1.0f;

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
