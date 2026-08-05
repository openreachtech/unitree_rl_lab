#pragma once

#include "FSM/State_RLBase.h"
#include "GaitMode.h"

#include <functional>
#include <string>
#include <vector>

/**
 * @brief RL state for a Go2 gait_mode-aware policy (e.g. Isaac Lab task `Go2-Biped-Phase1`).
 *
 * The network takes a discrete `gait_mode` one-hot observation alongside the usual
 * proprioception. Training currently pins that input to a single mode for the whole
 * episode (see `PinnedGaitModeCommand`) rather than resampling it, so the policy has
 * only ever seen `default_gait_mode` (hind_biped unless overridden in config.yaml).
 * The operator can still switch modes live -- through the joystick / keyboard bindings
 * under `FSM.<state>.gait_modes` and `FSM.<state>.keyboard_gait_modes` -- for exploring
 * how a pinned-mode checkpoint responds to a mode it was never trained to expect, but
 * only `default_gait_mode` is validated behavior.
 */
class State_BipedRL : public State_RLBase
{
public:
    State_BipedRL(int state_mode, std::string state_string);

    void enter() override;
    void run() override;

protected:
    // The biped stances hold the base pitched ~70-90 degrees off flat, which the
    // quadruped tilt limit reads as a fall, so that limit only applies once the robot
    // is actually standing on four legs again -- see update_tilt_limit_arming().
    bool fall_detected() const override;

private:
    struct ModeTrigger
    {
        std::function<bool()> triggered;
        go2::GaitMode mode;
    };

    void load_mode_triggers(YAML::Node cfg, const std::string& state_string);
    void update_tilt_limit_arming();

    std::vector<ModeTrigger> mode_triggers_;

    // Mode commanded to the policy on entry, before any operator trigger fires.
    // Config key: FSM.<state>.default_gait_mode (name string, e.g. "hind_biped").
    go2::GaitMode default_gait_mode_ = go2::GaitMode::kHindBiped;

    // Only meant to catch a full flip; see fall_detected().
    float biped_tilt_limit_ = 2.6;

    // Whether the strict quadruped tilt limit is in force. Commanding quad only starts
    // the robot lowering itself; the limit is armed when it has actually settled.
    bool quad_tilt_limit_armed_ = false;
};

REGISTER_FSM(State_BipedRL)
