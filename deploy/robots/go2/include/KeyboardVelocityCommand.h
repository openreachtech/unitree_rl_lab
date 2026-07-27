#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

namespace go2
{

// Latched driving command for controllers without a wireless remote, backing the
// `keyboard_velocity_commands` observation.
//
// deploy.yaml:
//   observations: keyboard_velocity_commands  (not velocity_commands)
//   commands.base_velocity:
//     keyboard_vel_scale: 0.8   # optional, default 0.8
//     keyboard_alpha: 0.15      # optional low-pass; higher = smoother / slower
//
// The biped policies list the command twice -- once for the current step and once for
// the stacked history block -- so the low-pass filter has to advance once per env step
// instead of once per observation term. The result is therefore cached on
// (env, episode_length): a step counter that has not advanced returns the cached
// command, and one that went backwards means the state was re-entered, which clears
// the filter as well.
inline std::vector<float> keyboard_velocity_command(isaaclab::ManagerBasedRLEnv* env)
{
    static const isaaclab::ManagerBasedRLEnv* owner = nullptr;
    static long cached_step = -1;
    static std::array<float, 3> cmd = {0.0f, 0.0f, 0.0f};

    if (owner == env && cached_step == env->episode_length)
    {
        return std::vector<float>(cmd.begin(), cmd.end());
    }
    if (owner != env || env->episode_length < cached_step)
    {
        cmd = {0.0f, 0.0f, 0.0f};
    }
    owner = env;
    cached_step = env->episode_length;

    if (!FSMState::keyboard)
    {
        FSMState::keyboard = std::make_shared<Keyboard>();
    }
    auto keyboard = FSMState::keyboard;

    const auto cmd_cfg = env->cfg["commands"]["base_velocity"];
    const auto ranges = cmd_cfg["ranges"];

    float vel_scale = 0.8f;
    if (cmd_cfg["keyboard_vel_scale"].IsDefined())
    {
        vel_scale = cmd_cfg["keyboard_vel_scale"].as<float>();
    }

    float alpha = 0.15f;
    if (cmd_cfg["keyboard_alpha"].IsDefined())
    {
        alpha = cmd_cfg["keyboard_alpha"].as<float>();
    }

    const auto sx = [&](int idx) { return vel_scale * ranges["lin_vel_x"][idx].as<float>(); };
    const auto sy = [&](int idx) { return vel_scale * ranges["lin_vel_y"][idx].as<float>(); };
    const auto sz = [&](int idx) { return vel_scale * ranges["ang_vel_z"][idx].as<float>(); };

    if (keyboard->consume_velocity_stop())
    {
        cmd = {0.0f, 0.0f, 0.0f};
        return std::vector<float>(cmd.begin(), cmd.end());
    }

    std::array<float, 3> target = {0.0f, 0.0f, 0.0f};
    if (keyboard->pressed("f"))
    {
        target[0] += sx(1);
    }
    if (keyboard->pressed("b"))
    {
        target[0] += sx(0);
    }
    if (keyboard->pressed("l"))
    {
        target[1] += sy(1);
    }
    if (keyboard->pressed("r"))
    {
        target[1] += sy(0);
    }
    if (keyboard->pressed("y"))
    {
        target[2] += sz(1);
    }
    if (keyboard->pressed("u"))
    {
        target[2] += sz(0);
    }

    for (int i = 0; i < 3; ++i)
    {
        cmd[i] = (1.0f - alpha) * cmd[i] + alpha * target[i];
    }

    cmd[0] = std::clamp(cmd[0], sx(0), sx(1));
    cmd[1] = std::clamp(cmd[1], sy(0), sy(1));
    cmd[2] = std::clamp(cmd[2], sz(0), sz(1));

    return std::vector<float>(cmd.begin(), cmd.end());
}

} // namespace go2
