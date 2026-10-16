// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#include "State_Biped.h"
#include "BipedLogger.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"

#include <chrono>

std::atomic<float> State_Biped::active_stance{1.0f};
std::mutex State_Biped::command_mutex_;
std::vector<float> State_Biped::command_(3, 0.0f);

std::vector<float> State_Biped::last_command()
{
    std::lock_guard<std::mutex> lock(command_mutex_);
    return command_;
}

void State_Biped::record_command(const std::vector<float> & command)
{
    std::lock_guard<std::mutex> lock(command_mutex_);
    command_ = command;
}

namespace isaaclab
{
namespace mdp
{

// handstand_command : [enabled, stance]. Mirrors HandstandCommand.command (handstand.py); the
// stance is held for as long as the state is active, as in training.
REGISTER_OBSERVATION(handstand_command)
{
    (void)env;
    (void)params;
    return std::vector<float>{1.0f, State_Biped::active_stance.load()};
}

// The keyboard drive (keyboard_velocity_commands, State_RLBase.cpp), recorded for the log.
REGISTER_OBSERVATION(biped_velocity_commands)
{
    auto command = observations_map().at("keyboard_velocity_commands")(env, params);
    State_Biped::record_command(command);
    return command;
}

} // namespace mdp
} // namespace isaaclab

namespace
{

// Rename observations.velocity_commands -> biped_velocity_commands, keeping every term in its
// place: the order of the observation terms is the ONNX input layout.
YAML::Node with_keyboard_velocity(YAML::Node deploy_cfg)
{
    auto observations = deploy_cfg["observations"];
    if (!observations || !observations["velocity_commands"])
    {
        return deploy_cfg;
    }
    YAML::Node renamed(YAML::NodeType::Map);
    for (const auto & term : observations)
    {
        const auto name = term.first.as<std::string>();
        renamed[name == "velocity_commands" ? "biped_velocity_commands" : name] = term.second;
    }
    deploy_cfg["observations"] = renamed;
    return deploy_cfg;
}

} // namespace

State_Biped::State_Biped(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    if (cfg["stance"]) stance_ = cfg["stance"].as<float>();
    if (cfg["tilt_limit"]) tilt_limit_ = cfg["tilt_limit"].as<float>();

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        with_keyboard_velocity(YAML::LoadFile(policy_dir / "params" / "deploy.yaml")),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");
    action_map_ = std::make_unique<JointActionMap>(env->cfg);

    // Same keys as the Go2 handstand controller on feat/biped, plus log_dir.
    BipedLogger::Config log_cfg;
    if (cfg["log_dir"])      log_cfg.log_dir = cfg["log_dir"].as<std::string>();
    if (cfg["telemetry"])    log_cfg.telemetry = cfg["telemetry"].as<bool>();
    if (cfg["torque_log"])   log_cfg.torque_log = cfg["torque_log"].as<bool>();
    if (cfg["torque_log_s"]) log_cfg.torque_log_s = cfg["torque_log_s"].as<float>();
    logger_ = std::make_unique<BipedLogger>(log_cfg, state_string, FSMState::lowstate, FSMState::lowcmd.get());

    spdlog::info("State_{}: stance {:+.0f} ({}), tilt limit {:.2f} rad, policy {}",
                 state_string, stance_, stance_ > 0 ? "front wheels" : "hind wheels",
                 tilt_limit_, policy_dir.string());

    this->registered_checks.emplace_back(
        std::make_pair(
            [this]() -> bool { return isaaclab::mdp::bad_orientation(env.get(), tilt_limit_); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_Biped::enter()
{
    // deploy.yaml stores stiffness/damping already in SDK motor order (export_deploy_cfg).
    for (size_t i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        auto & motor = lowcmd->msg_.motor_cmd()[i];
        motor.kp() = env->robot->data.joint_stiffness[i];
        motor.kd() = env->robot->data.joint_damping[i];
        motor.dq() = 0;
        motor.tau() = 0;
    }

    active_stance.store(stance_);
    record_command({0.0f, 0.0f, 0.0f});
    logger_->on_enter(stance_, env->step_dt);
    env->robot->update();

    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        auto sleepTill = clock::now() + dt;
        env->reset();

        while (policy_thread_running)
        {
            const auto work_start = clock::now();
            env->step();
            logger_->on_policy_step(
                last_command(), std::chrono::duration<float, std::milli>(clock::now() - work_start).count());
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_Biped::run()
{
    action_map_->write(env->action_manager->processed_actions(), *lowcmd);
    // After the targets are written, so tau_cmd pairs the command in force with the state it acts on.
    logger_->on_fsm_tick();
}

void State_Biped::exit()
{
    policy_thread_running = false;
    if (policy_thread.joinable())
    {
        policy_thread.join();
    }
    // After the join: no telemetry row can be in flight when the stream closes.
    logger_->on_exit();
}

State_Biped::~State_Biped() = default;
