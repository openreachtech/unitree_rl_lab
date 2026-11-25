#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/ros2/String_.hpp>

// unitree_mujoco's elastic band, driven from this terminal (sim only; no subscriber on
// the real robot, so the messages go nowhere there).
static constexpr const char* ELASTIC_BAND_TOPIC = "rt/sim/elastic_band";

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

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

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Remote:   [L2+Up] FixStand, [R1+X] Velocity, [L2+B] Passive\n";
    std::cout << "Keyboard: [1] FixStand, [Enter] Velocity, [0] Passive\n";
    std::cout << "          [f/b/l/r/y/u] forward/back/left/right/turn-left/turn-right (latched, combine OK)\n";
    std::cout << "          [Space] zero velocity command\n";
    std::cout << "Sim band: [8] lower, [7] raise, [9] release / re-attach (unitree_mujoco -e 1)\n";

    unitree::robot::ChannelPublisher<std_msgs::msg::dds_::String_> band_pub(ELASTIC_BAND_TOPIC);
    band_pub.InitChannel();
    const std::pair<const char*, const char*> band_keys[] = {{"9", "toggle"}, {"8", "down"}, {"7", "up"}};

    while (true)
    {
        for (const auto& [key, cmd] : band_keys)
        {
            for (int n = FSMState::keyboard->consume_presses(key); n > 0; --n)
            {
                std_msgs::msg::dds_::String_ msg;
                msg.data(cmd);
                band_pub.Write(msg);
                spdlog::info("Sim band: {}", cmd);
            }
        }
        usleep(20000);
    }
    
    return 0;
}

