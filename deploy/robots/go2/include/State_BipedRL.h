#pragma once

#include "FSM/State_RLBase.h"
#include "GaitMode.h"

#include <functional>
#include <string>
#include <vector>

/**
 * @brief RL state for the Go2 multimode policy (Isaac Lab task `Unitree-Go2-Multimode`).
 *
 * One network walks the robot either as a quadruped or on a single leg pair. Training
 * drives that choice with a discrete `gait_mode` command that resamples mid-episode;
 * here the operator drives it instead, through the joystick / keyboard bindings under
 * `FSM.<state>.gait_modes` and `FSM.<state>.keyboard_gait_modes`, and the selection is
 * handed to the policy as the same one-hot observation.
 */
class State_BipedRL : public State_RLBase
{
public:
    State_BipedRL(int state_mode, std::string state_string);

    void enter() override;
    void run() override;

protected:
    // The biped stances hold the base pitched ~70-90 degrees off flat, which the
    // quadruped tilt limit reads as a fall, so that limit only applies in quad mode.
    bool fall_detected() const override;

private:
    struct ModeTrigger
    {
        std::function<bool()> triggered;
        go2::GaitMode mode;
    };

    void load_mode_triggers(YAML::Node cfg, const std::string& state_string);

    std::vector<ModeTrigger> mode_triggers_;

    // Only meant to catch a full flip; see fall_detected().
    float biped_tilt_limit_ = 2.6;
};

REGISTER_FSM(State_BipedRL)
