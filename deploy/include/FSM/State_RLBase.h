// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

#include <atomic>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

        policy_step_max_ms_.store(0.0f);
        policy_step_last_ms_.store(0.0f);
        policy_overrun_count_.store(0);

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                const auto work_start = clock::now();

                env->step();
                on_policy_step();

                const float work_ms =
                    std::chrono::duration<float, std::milli>(clock::now() - work_start).count();
                policy_step_last_ms_.store(work_ms);
                if (work_ms > policy_step_max_ms_.load())
                {
                    policy_step_max_ms_.store(work_ms);
                }
                if (work_ms > env->step_dt * 1e3f)
                {
                    policy_overrun_count_.fetch_add(1);
                }

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

protected:
    // Bail-out condition back to Passive. Overridable for policies whose target
    // posture is itself a large tilt, where a single fixed limit cannot tell the
    // intended stance from a fall (see go2's State_BipedRL).
    virtual bool fall_detected() const
    {
        return isaaclab::mdp::bad_orientation(env.get(), tilt_limit_);
    }

    // Called on the policy thread immediately after each env->step(), i.e. once per
    // control step with the action the FSM thread is about to publish. Default no-op;
    // overridden by states that record per-step diagnostics.
    virtual void on_policy_step() {}

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    // Tilt of the base away from vertical [rad] treated as a fall.
    float tilt_limit_ = 1.0;

    // Policy-loop health. The loop targets step_dt per iteration but runs ONNX
    // inference -- and, for a state that overrides on_policy_step(), a file write --
    // inline. Once an iteration overruns, sleep_until stops sleeping and the effective
    // control rate silently drops; that cannot happen in simulation, so it is measured
    // rather than assumed. Written by the policy thread, read by the FSM thread.
    std::atomic<float> policy_step_max_ms_{0.0f};
    std::atomic<float> policy_step_last_ms_{0.0f};
    std::atomic<int> policy_overrun_count_{0};

private:
    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
