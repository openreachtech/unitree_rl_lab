// Get-up FSM state for the policies of tasks/locomotion/robots/g1_getup (HoST-style get-up).
//
// Differs from State_RLBase where the get-up task's training differs from a locomotion task:
// * delta action: the motor target is q_now + scale * a, rebuilt from the measured joint position
//   at every 1 kHz control tick (the training rebuilt it at every physics step), so the motor PD
//   gives ~ kp * scale * a - kd * dq;
// * for the first `unactuated_steps` policy steps (0.6 s) the joints are held limp (target = q_now)
//   and the policy sees all-zero observations, as in training;
// * no bad-orientation exit to Passive: the state starts with the robot lying on the ground;
// * motors that the policy does not drive (on the 23dof: waist roll / pitch, wrist pitch / yaw,
//   which exist in the 29-joint simulation model) are held at 0 with `hold_kp` / `hold_kd`.

#pragma once

#include <atomic>
#include <set>
#include <mutex>

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/delta_joint_actions.h"

class State_GetUp : public FSMState
{
public:
    State_GetUp(int state_mode, std::string state_string);

    void enter()
    {
        auto & data = env->robot->data;
        // policy joints: training gains from deploy.yaml (sdk order)
        for (int i = 0; i < (int)data.joint_stiffness.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = data.joint_stiffness[i];
            motor.kd() = data.joint_damping[i];
            motor.dq() = 0;
            motor.tau() = 0;
            motor.q() = lowstate->msg_.motor_state()[i].q();
        }
        // motors outside the policy: hold at 0
        for (int m = 0; m < num_motors_; ++m)
        {
            if (policy_motors_.count(m)) continue;
            auto & motor = lowcmd->msg_.motor_cmd()[m];
            motor.kp() = hold_kp_;
            motor.kd() = hold_kd_;
            motor.q() = 0;
            motor.dq() = 0;
            motor.tau() = 0;
        }

        env->robot->update();
        actuated_ = false;
        {
            std::lock_guard<std::mutex> lock(delta_mutex_);
            delta_.assign(data.joint_ids_map.size(), 0.0f);
        }

        policy_thread_running_ = true;
        policy_thread_ = std::thread([this] {
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desired(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desired);
            auto sleep_till = clock::now() + dt;

            env->observation_manager->zero_observations = unactuated_steps_ > 0;
            env->reset();
            while (policy_thread_running_)
            {
                // training: the observation for the action of step k is zeroed while k < unactuated
                // steps, and that action drives the joints only from step k = unactuated steps on
                long k = env->episode_length;  // step index of the action computed next
                env->observation_manager->zero_observations = k < unactuated_steps_;
                env->step();
                {
                    std::lock_guard<std::mutex> lock(delta_mutex_);
                    delta_ = env->action_manager->processed_actions();
                }
                if (k >= unactuated_steps_ && !actuated_)
                {
                    spdlog::info("GetUp: policy now driving the joints");
                }
                actuated_ = k >= unactuated_steps_;

                std::this_thread::sleep_until(sleep_till);
                sleep_till += dt;
            }
        });
        spdlog::info("GetUp: limp for {} steps ({:.2f} s), then the policy", unactuated_steps_,
                     unactuated_steps_ * env->step_dt);
    }

    void run()
    {
        std::vector<float> delta;
        {
            std::lock_guard<std::mutex> lock(delta_mutex_);
            delta = delta_;
        }
        const bool actuated = actuated_;
        auto & ids = env->robot->data.joint_ids_map;
        for (int i = 0; i < (int)ids.size(); ++i)
        {
            int m = (int)ids[i];
            float q_now = lowstate->msg_.motor_state()[m].q();
            lowcmd->msg_.motor_cmd()[m].q() = actuated ? q_now + delta[i] : q_now;
        }
    }

    void exit()
    {
        policy_thread_running_ = false;
        if (policy_thread_.joinable()) policy_thread_.join();
        env->observation_manager->zero_observations = false;
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread_;
    std::atomic<bool> policy_thread_running_{false};
    std::atomic<bool> actuated_{false};
    std::mutex delta_mutex_;
    std::vector<float> delta_;

    int unactuated_steps_ = 0;
    int num_motors_ = 0;
    std::set<int> policy_motors_;
    float hold_kp_ = 100.0f;
    float hold_kd_ = 5.0f;
};

REGISTER_FSM(State_GetUp)
