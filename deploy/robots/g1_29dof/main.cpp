#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "FSM/State_GetUp.h"
#include "State_Mimic.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

// Drop enabled RL states whose policy directory does not exist, so that the controller still starts
// (e.g. GetUp alone on a machine without the velocity policy).
void drop_states_without_policy()
{
    auto fsms = param::config["FSM"]["_"];
    std::vector<std::string> missing;
    for (auto it = fsms.begin(); it != fsms.end(); ++it)
    {
        auto name = it->first.as<std::string>();
        auto dir = param::config["FSM"][name]["policy_dir"];
        if (!dir) continue;
        std::filesystem::path p = dir.as<std::string>();
        if (p.is_relative()) p = param::proj_dir / p;
        if (!std::filesystem::exists(p)) missing.push_back(name);
    }
    for (auto & name : missing)
    {
        spdlog::warn("FSM State_{} disabled: policy_dir not found ({})", name,
                     param::config["FSM"][name]["policy_dir"].as<std::string>());
        fsms.remove(name);
    }
}

// Keyboard transitions: `keys: {TargetState: "<key>"}` under each state in config.yaml, read from the
// terminal running g1_ctrl (for MuJoCo without a gamepad).
void add_keyboard_transitions(CtrlFSM & fsm)
{
    for (auto & state : fsm.states)
    {
        auto keys = param::config["FSM"][state->getStateString()]["keys"];
        if (!keys) continue;
        for (auto it = keys.begin(); it != keys.end(); ++it)
        {
            auto target = it->first.as<std::string>();
            auto key = it->second.as<std::string>();
            if (!FSMStringMap.right.count(target)) continue;
            int id = FSMStringMap.right.at(target);
            state->registered_checks.emplace_back(
                [key]() -> bool { return FSMState::keyboard && FSMState::keyboard->on_pressed && FSMState::keyboard->key() == key; },
                id);
        }
    }
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    // DDS domain: 0 unless UNITREE_DDS_DOMAIN is set (keeps a local MuJoCo test off other users' domain 0)
    int domain_id = std::getenv("UNITREE_DDS_DOMAIN") ? std::atoi(std::getenv("UNITREE_DDS_DOMAIN")) : 0;
    unitree::robot::ChannelFactory::Instance()->Init(domain_id, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    drop_states_without_policy();
    FSMState::keyboard = std::make_shared<Keyboard>();
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    add_keyboard_transitions(*fsm);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";
    std::cout << "Get-up: [L2 + Down] from Passive. Keyboard (this terminal): 0 Passive, 1 FixStand, 2 Velocity, 3 GetUp.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

