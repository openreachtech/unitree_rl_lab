#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <array>
#include <algorithm>
#include <utility>

namespace isaaclab
{

// Keyboard drive, ported from deploy/robots/go2. Requires the policy's deploy.yaml to
// name this observation instead of the stock velocity_commands, which train.py does when
// passed --deploy-keyboard-commands (it rewrites the key in place, preserving observation
// order -- the order is part of the ONNX input layout, so it must not shift).
//
// deploy.yaml:
//   observations: keyboard_velocity_commands  (not velocity_commands)
//   commands.base_velocity:
//     keyboard_vel_scale: 0.8   # optional, default 0.8
//     keyboard_alpha: 0.15      # optional low-pass; higher = smoother / slower
//
// Note on ranges: export_deploy_cfg writes commands.base_velocity.ranges from the env's
// *limit_ranges* when it has them, so the keys below span the widest command the policy
// was ever trained to accept, not whatever the curriculum had reached at export time.
// A one-sided range therefore yields a one-sided key: under the Phase5 configs, for
// instance, lin_vel_x is (0.4, 1.2) and lin_vel_y is (0, 0), so [b] still commands
// forward motion and [l]/[r] do nothing. That is faithful to the policy -- it was never
// trained to reverse or strafe -- not a bug in this mapping.

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

    // Neutral has to stay reachable. These bounds come from the policy's limit_ranges,
    // and Phase5's are one-sided (lin_vel_x = 0.4 .. 1.2 -- the non-zero floor is
    // deliberate, see CommandsCfgPhase5), so clamping to [sx(0), sx(1)] alone would pin
    // the command at 0.8 * 0.4 = 0.32 m/s with nothing pressed and the robot would drive
    // off on its own. (Space still zeroed it for exactly one control step, because
    // consume_velocity_stop returns before this clamp, then the clamp pulled it back up.)
    // Widening each bound to include zero fixes that and is a no-op for a symmetric range
    // like Go2's, where zero already sat inside the interval.
    const auto with_zero = [](float lo, float hi) {
        return std::pair<float, float>(std::min(0.0f, lo), std::max(0.0f, hi));
    };
    const auto bx = with_zero(sx(0), sx(1));
    const auto by = with_zero(sy(0), sy(1));
    const auto bz = with_zero(sz(0), sz(1));
    cmd[0] = std::clamp(cmd[0], bx.first, bx.second);
    cmd[1] = std::clamp(cmd[1], by.first, by.second);
    cmd[2] = std::clamp(cmd[2], bz.first, bz.second);

    return std::vector<float>(cmd.begin(), cmd.end());
}

} // namespace isaaclab

namespace
{

// Action index -> SDK motor id, resolved once in the constructor (see run() for why this
// composition is needed rather than joint_ids_map alone).
std::vector<int> g_pos_motor_ids;
std::vector<int> g_vel_motor_ids;

std::vector<int> resolve_motor_ids(YAML::Node action_cfg, const std::vector<int>& joint_ids_map)
{
    std::vector<int> motor_ids;
    auto joint_ids_node = action_cfg["joint_ids"];
    if (!joint_ids_node || joint_ids_node.IsNull())
    {
        // No explicit selection: the term spans every joint in IsaacLab order, so the
        // action index *is* the IsaacLab joint id.
        return joint_ids_map;
    }
    for (int isaac_joint_id : joint_ids_node.as<std::vector<int>>())
    {
        motor_ids.push_back(joint_ids_map.at(isaac_joint_id));
    }
    return motor_ids;
}

} // namespace

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

    const auto joint_ids_map = env->cfg["joint_ids_map"].as<std::vector<int>>();
    g_pos_motor_ids = resolve_motor_ids(env->cfg["actions"]["JointPositionAction"], joint_ids_map);
    g_vel_motor_ids = resolve_motor_ids(env->cfg["actions"]["JointVelocityAction"], joint_ids_map);

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}
void State_RLBase::run()
{
    // processed_actions() concatenates the action terms in deploy.yaml order (position
    // then velocity here), and element k of a term refers to that term's joint_ids[k] --
    // an *IsaacLab* joint id. joint_ids_map is itself indexed by IsaacLab joint id, so the
    // SDK motor is joint_ids_map[joint_ids[k]]; indexing it with the action index k
    // directly is only correct when a term's joint_ids happen to be the identity.
    //
    // That identity holds on Go2, whose sole action term is joint_names=[".*"], which is
    // why the equivalent loop there is right. It does *not* hold on Go2W: the legs and
    // wheels need separate position/velocity terms, so both declare explicit
    // SDK-ordered joint_names with preserve_order=True and their joint_ids come out as
    // permutations ([1,5,9,0,...] and [13,12,15,14]). Skipping the composition therefore
    // cross-wired 14 of the 16 joints -- FR_thigh's target was driving FR_hip, FR_calf's
    // was driving RL_hip, and the left/right wheels were swapped -- which made the robot
    // thrash the instant the policy took over.
    auto action = env->action_manager->processed_actions();

    for(size_t i = 0; i < g_pos_motor_ids.size(); i++) {
        lowcmd->msg_.motor_cmd()[g_pos_motor_ids[i]].q() = action[i];
    }
    const size_t vel_offset = g_pos_motor_ids.size();
    for(size_t i = 0; i < g_vel_motor_ids.size(); i++) {
        lowcmd->msg_.motor_cmd()[g_vel_motor_ids[i]].dq() = action[vel_offset + i];
    }
}