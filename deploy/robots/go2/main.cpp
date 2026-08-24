#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Flip.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
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

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::keyboard = std::make_shared<Keyboard>();

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Remote: [L2+A] FixStand, [Start] Velocity, [L2+Up] Dynamic, [L2+B] Passive\n";
    std::cout << "Keyboard: [1] FixStand, [Enter] Velocity, [2] Dynamic, [0] Passive\n";
    std::cout << "          [F/B/L/R/Y/U] velocity forward/back/left/right/yaw (latched, diagonal OK)\n";
    std::cout << "          [Space] zero velocity cmd; [0] Passive FSM\n";
    std::cout << "Dynamic: enter from FixStand, then [3] jump, [4] backflip, [5] sideflip (repeatable);\n";
    std::cout << "         [1] back to FixStand, [0] Passive.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

