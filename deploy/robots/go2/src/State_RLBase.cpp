#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "HeightScanUpdater.h"
#include "param.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <random>
#include <utility>
#include <spdlog/spdlog.h>

namespace isaaclab
{

namespace
{

std::vector<float> apply_height_scan_noise(
    std::vector<float> scan,
    const YAML::Node& noise_cfg)
{
    if (!noise_cfg.IsDefined() ||
        !noise_cfg["enabled"].IsDefined() ||
        !noise_cfg["enabled"].as<bool>())
    {
        return scan;
    }

    float noise_min = -0.05f;
    float noise_max = 0.05f;
    std::uint32_t seed = 0;
    if (noise_cfg["min"].IsDefined())
    {
        noise_min = noise_cfg["min"].as<float>();
    }
    if (noise_cfg["max"].IsDefined())
    {
        noise_max = noise_cfg["max"].as<float>();
    }
    if (noise_cfg["seed"].IsDefined())
    {
        seed = noise_cfg["seed"].as<std::uint32_t>();
    }
    if (noise_min > noise_max)
    {
        std::swap(noise_min, noise_max);
    }

    static std::mt19937 generator;
    static std::uint32_t generator_seed = 0;
    static bool initialized = false;
    if (!initialized || generator_seed != seed)
    {
        generator.seed(seed);
        generator_seed = seed;
        initialized = true;
    }

    std::uniform_real_distribution<float> distribution(noise_min, noise_max);
    for (auto& value : scan)
    {
        value += distribution(generator);
    }

    static bool logged = false;
    if (!logged)
    {
        spdlog::info(
            "height_scan: uniform noise ON (range [{:.4f}, {:.4f}], seed {})",
            noise_min,
            noise_max,
            seed);
        logged = true;
    }
    return scan;
}

} // namespace

// deploy.yaml:
//   observations: keyboard_velocity_commands  (not velocity_commands)
//   commands.base_velocity:
//     keyboard_vel_scale: 0.8   # optional, default 0.8
//     keyboard_alpha: 0.15      # optional low-pass; higher = smoother / slower

REGISTER_OBSERVATION(keyboard_velocity_commands)
{
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

    static std::array<float, 3> cmd = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> target = {0.0f, 0.0f, 0.0f};

    if (keyboard->consume_velocity_stop())
    {
        cmd = {0.0f, 0.0f, 0.0f};
        return std::vector<float>(cmd.begin(), cmd.end());
    }

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

// deploy.yaml must list observations.height_scan (exported from policy training).
// Optional debug overrides (constant flat terrain for the policy):
//   observations.height_scan.params.flat_override / flat_value
//   OR config.yaml FSM.Velocity.height_scan.flat_override / flat_value
// Optional config.yaml uniform noise:
//   FSM.Velocity.height_scan.noise: {enabled: false, min: -0.05, max: 0.05, seed: 0}
REGISTER_OBSERVATION(height_scan)
{
    (void)env;

    bool flat_override = false;
    float flat_value = go2::kHeightScanFlatDefault;

    if (params["flat_override"].IsDefined() && params["flat_override"].as<bool>())
    {
        flat_override = true;
        if (params["flat_value"].IsDefined())
        {
            flat_value = params["flat_value"].as<float>();
        }
    }

    // config.yaml takes precedence over deploy.yaml and can force flat_override on or off.
    const auto fsm_height_scan = param::config["FSM"]["Velocity"]["height_scan"];
    if (fsm_height_scan.IsDefined() && fsm_height_scan["flat_override"].IsDefined())
    {
        flat_override = fsm_height_scan["flat_override"].as<bool>();
        if (flat_override && fsm_height_scan["flat_value"].IsDefined())
        {
            flat_value = fsm_height_scan["flat_value"].as<float>();
        }
    }

    std::vector<float> scan;
    if (flat_override)
    {
        static bool logged = false;
        if (!logged)
        {
            spdlog::info(
                "height_scan: flat_override ON (policy gets constant {:.4f}, DDS pipeline unchanged)",
                flat_value);
            logged = true;
        }
        scan = go2::make_flat_height_scan(flat_value);
    }
    else
    {
        scan = go2::HeightScanUpdater::instance().get();
    }

    return apply_height_scan_noise(std::move(scan), fsm_height_scan["noise"]);
}

} // namespace isaaclab

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
