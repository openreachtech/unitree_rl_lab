#include "FSM/FSMState.h"
#include "isaaclab/manager/observation_manager.h"
#include <spdlog/spdlog.h>
#include <memory>

namespace isaaclab
{

namespace
{
void ensure_keyboard()
{
    if (!FSMState::keyboard)
    {
        FSMState::keyboard = std::make_shared<Keyboard>();
        spdlog::info(
            "Keyboard control enabled. FSM: [1] FixStand, [Enter] Velocity, [0] Passive. "
            "Velocity: W/S forward-back, A/D strafe, Q/E yaw. "
            "In policy params/deploy.yaml use observation 'keyboard_velocity_commands' "
            "instead of 'velocity_commands'.");
    }
}
} // namespace

// In <policy_dir>/params/deploy.yaml, replace observation key:
//   velocity_commands  ->  keyboard_velocity_commands
// (keep params/clip/scale/history_length unchanged)
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    ensure_keyboard();

    std::string key = FSMState::keyboard->key();
    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    // cmd = [lin_vel_x, lin_vel_y, ang_vel_z]; each key snaps to range max/min (full stick).
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key == "w") // forward
    {
        cmd = {cfg["lin_vel_x"][1].as<float>(), 0.0f, 0.0f};
    }
    else if (key == "s") // backward
    {
        cmd = {cfg["lin_vel_x"][0].as<float>(), 0.0f, 0.0f};
    }
    else if (key == "a") // strafe left
    {
        cmd = {0.0f, cfg["lin_vel_y"][1].as<float>(), 0.0f};
    }
    else if (key == "d") // strafe right
    {
        cmd = {0.0f, cfg["lin_vel_y"][0].as<float>(), 0.0f};
    }
    else if (key == "q") // yaw left (CCW)
    {
        cmd = {0.0f, 0.0f, cfg["ang_vel_z"][1].as<float>()};
    }
    else if (key == "e") // yaw right (CW)
    {
        cmd = {0.0f, 0.0f, cfg["ang_vel_z"][0].as<float>()};
    }

    return cmd;
}

} // namespace isaaclab
