#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <array>
#include <string>
#include <algorithm>

#include <spdlog/spdlog.h>

namespace isaaclab
{

// deploy.yaml:
//   observations: keyboard_velocity_commands  (not velocity_commands)
//   commands.base_velocity:
//     keyboard_vel_scale: 0.5   # optional, default 0.5
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

    float vel_scale = 0.5f;
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

    // The clamp must always admit 0. "Stop" is a deploy-side concept and exists
    // independently of the training command distribution, but these bounds come from
    // commands.base_velocity.ranges. Loop 10 narrowed lin_vel_x to U(2.8, 3.3), which
    // made sx(0) = 2.8 a *floor*: entering Velocity immediately drove the robot at
    // 2.8 m/s, and space could not stop it -- consume_velocity_stop() zeroed cmd for
    // exactly one control step and this clamp put it straight back on the next one.
    // Widening each interval to include zero keeps the reachable command set equal to
    // the trained range plus the stop, whatever sign the configured bounds have.
    const auto clamp_with_stop = [](float v, float lo, float hi) {
        return std::clamp(v, std::min(lo, 0.0f), std::max(hi, 0.0f));
    };
    cmd[0] = clamp_with_stop(cmd[0], sx(0), sx(1));
    cmd[1] = clamp_with_stop(cmd[1], sy(0), sy(1));
    cmd[2] = clamp_with_stop(cmd[2], sz(0), sz(1));

    return std::vector<float>(cmd.begin(), cmd.end());
}

// deploy.yaml:
//   observations: jump_command
//   commands.jump_command:
//     jump_hold_time_s: 1.0   # optional, default 1.0
//
// This is the deploy-side counterpart of mdp.commands.jump_command.JumpCommand
// (see リファレンス_ExplicitEstimator実装仕様.md): 'j' triggers 0->1
// (Keyboard::consume_jump_trigger(), a one-shot exchange like
// consume_velocity_stop() but which does NOT clear the held velocity
// keys -- the approach run keeps going through the jump).
//
// KNOWN SIMPLIFICATION vs. training: JumpCommand resets to 0 on a detected
// landing (all four feet in contact + upright + minimum air time elapsed).
// This deploy layer has no per-foot contact/force sensing plumbed in yet
// (unlike IsaacSim's ContactSensor -- see unitree_articulation.h), so it
// resets after a fixed jump_hold_time_s instead. Revisit once foot-force
// sensing is available here; until then a jump attempt that lands
// meaningfully earlier or later than jump_hold_time_s will see the trained
// jump_command bit behave slightly differently than in sim.
// Shared between the jump_command and jump_time observations below. deploy.yaml
// lists jump_command first (it is declared first in PolicyCfgLongJump), so the
// command handler advances this state and jump_time then reads it in the same
// control step.
static float g_jump_cmd = 0.0f;
static float g_jump_time_since_trigger = 0.0f;
// Wall clock since the last trigger, in contrast to g_jump_time_since_trigger, which
// freezes when the command bit times out. The orientation relax window below is measured
// against this one: the flight can outlast the fixed jump_hold_time_s window, and the
// tilt check must not re-arm in mid-air just because the bit expired. Negative = no jump
// has been triggered since the controller started.
static float g_jump_since_trigger_wall = -1.0f;

REGISTER_OBSERVATION(jump_command)
{
    if (!FSMState::keyboard)
    {
        FSMState::keyboard = std::make_shared<Keyboard>();
    }
    auto keyboard = FSMState::keyboard;

    const auto cmd_cfg = env->cfg["commands"]["jump_command"];
    float hold_time_s = 1.0f;
    if (cmd_cfg["jump_hold_time_s"].IsDefined())
    {
        hold_time_s = cmd_cfg["jump_hold_time_s"].as<float>();
    }

    // Left-to-right && evaluation always consumes a pending trigger, even
    // if we are already mid-jump (cmd > 0) -- matching the training-time
    // JumpCommand, which likewise ignores (but still drains) a trigger
    // that arrives while an attempt is already in progress.
    if (keyboard->consume_jump_trigger() && g_jump_cmd == 0.0f)
    {
        g_jump_cmd = 1.0f;
        g_jump_time_since_trigger = 0.0f;
        g_jump_since_trigger_wall = 0.0f;
    }

    if (g_jump_since_trigger_wall >= 0.0f)
    {
        g_jump_since_trigger_wall += env->step_dt;
    }

    if (g_jump_cmd > 0.0f)
    {
        g_jump_time_since_trigger += env->step_dt;
        if (g_jump_time_since_trigger > hold_time_s)
        {
            g_jump_cmd = 0.0f;
        }
    }

    return std::vector<float>(1, g_jump_cmd);
}

// Mirror of mdp.jump_time_encoding (2026-09-02): normalised time since the jump
// trigger, 0 while not jumping. Added because the policy previously saw the jump
// as a bare 0/1 bit with no phase information -- it could not distinguish "load
// the legs now" from "you have been airborne 300 ms, prepare to land".
//
// The normaliser must match training's max_jump_duration_s (1.5 s), otherwise the
// deployed policy reads a differently-scaled phase than it trained on.
// export_deploy_cfg does not currently emit the jump command's own parameters
// into deploy.yaml (only commands.base_velocity.ranges), so the default below is
// the training value; it is still read from the config when present.
REGISTER_OBSERVATION(jump_time)
{
    const auto cmd_cfg = env->cfg["commands"]["jump_command"];
    float max_jump_duration_s = 1.5f;
    if (cmd_cfg["max_jump_duration_s"].IsDefined())
    {
        max_jump_duration_s = cmd_cfg["max_jump_duration_s"].as<float>();
    }

    float progress = 0.0f;
    if (g_jump_cmd > 0.0f && max_jump_duration_s > 0.0f)
    {
        progress = g_jump_time_since_trigger / max_jump_duration_s;
        progress = std::clamp(progress, 0.0f, 1.0f);
    }

    // Diagnostic trace (2026-09-06). The single tilt sample logged at the moment the FSM
    // cuts out only says the robot was already past the limit by then -- 1.50 to 2.95 rad
    // across four attempts, all reported at t=1.020 s. It cannot say WHEN the attitude went
    // wrong, and the two candidates want opposite fixes:
    //
    //   (i) the body already tumbles during the flight  -> the take-off imparts pitch
    //       angular momentum, which nothing in the air can undo (I_legs is 79% of the
    //       pitch inertia, but leg swing is a one-shot trade, not a brake).
    //   (ii) the attitude is fine until touchdown        -> it is a landing problem, and
    //       the leg-extension timing is what to fix.
    //
    // Logging the whole window at the 50 Hz policy rate separates them outright. Traced
    // once per jump for 1.5 s; costs nothing when no jump has been triggered.
    if (g_jump_since_trigger_wall >= 0.0f && g_jump_since_trigger_wall <= 1.5f)
    {
        const auto & g = env->robot->data.projected_gravity_b;
        const auto & w = env->robot->data.root_ang_vel_b;
        // Peak joint speed too: Isaac's actuator model derates torque above X1 = 13.5 rad/s
        // and reaches zero at X2 = 30, while mujoco's ctrlrange is flat in speed. If the
        // take-off never crosses 13.5 the two simulators agree on strength and that whole
        // hypothesis is dead; if it does, mujoco's knee is up to ~2x stronger exactly when
        // the pitch angular momentum is being set.
        const auto & dq = env->robot->data.joint_vel;
        int fastest = 0;
        float dq_max = 0.0f;
        for (int i = 0; i < dq.size(); ++i)
        {
            if (std::fabs(dq[i]) > dq_max) { dq_max = std::fabs(dq[i]); fastest = i; }
        }
        // Joint angles too (2026-09-09). The first pass logged only dq_max, which says how
        // hard a joint is moving but not what POSE the machine is in. tanaka's mujoco
        // observation -- "whether it lands is decided at the instant it leaves the ground"
        // -- was confirmed from the tilt trace (take-off tilt < 0.19 rad separated 20 of 21
        // attempts), but tilt is a body-attitude scalar and cannot say which leg produced
        // it. q is what distinguishes "the rear legs finished extending before the fronts
        // folded" from "the trunk was already pitched when the push began".
        // Order is IsaacLab's, grouped by joint type, per deploy.yaml default_joint_pos
        // [0.1 -0.1 0.1 -0.1 | 0.8 0.8 1.0 1.0 | -1.5 x4]:
        //   j0-3  hip   FL FR RL RR
        //   j4-7  thigh FL FR RL RR
        //   j8-11 calf  FL FR RL RR
        const auto & q = env->robot->data.joint_pos;
        std::string qs;
        for (int i = 0; i < q.size(); ++i)
        {
            qs += fmt::format("{}{:+.3f}", i ? " " : "", q[i]);
        }
        spdlog::info("JUMPTRACE t={:.3f} cmd={:.0f} phase={:.3f} tilt={:.3f} g=[{:+.3f} {:+.3f} {:+.3f}] "
                     "w=[{:+.2f} {:+.2f} {:+.2f}] dq_max={:.1f}(j{}) q=[{}]",
                     g_jump_since_trigger_wall, g_jump_cmd, progress,
                     std::fabs(std::acos(std::clamp(-g[2], -1.0f, 1.0f))),
                     g[0], g[1], g[2], w[0], w[1], w[2], dq_max, fastest, qs);
    }

    return std::vector<float>(1, progress);
}

// Deploy-side mirror of the ``relax_window_s`` gate in mdp.bad_orientation_grounded
// (terminations.py). Training tolerates a tilt past ``limit_angle`` for the first
// ``relax_window_s`` (1.0 s) after the jump trigger, and only while the command bit is
// still up -- without it, in the training code's own words, "the stock termination
// would end the episode on every attempt", because a jump pitches the body away from
// flat through the whole airborne phase and the landing recovery.
//
// The deploy check below had no such gate. Any jump that pitched past the limit
// switched the FSM straight to Passive, which is kp=0 / kd-only with q held at the
// current joint angles -- the robot goes limp *in the pose it was in* and topples. That
// is exactly what tanaka reported from mujoco on model_24600 (2026-09-06): "the height
// looks right, but it freezes in a strange pose in the air and falls over backwards",
// with no recovery afterwards because nothing leaves Passive on its own.
//
// This is プロジェクト_報酬項の名前と実測対象のズレ7例.md's pattern (2) again -- a
// training-side setting that never reached the robot -- and the fifth instance of it.
// The gate mirrors training exactly: relaxed only while jump_command is up AND within
// the window, so it ends the moment the jump window closes on landing.
// 2026-09-06, second pass. The first version gated on ``g_jump_cmd > 0`` to mirror
// training's ``jumping & within_relax_window`` literally. That mirror is wrong on the
// deploy side, because the two ``jumping`` bits do not mean the same thing:
//
//   training: the bit clears when landing is DETECTED, so "bit up" covers the whole
//             flight by construction, however long it lasts.
//   deploy:   the bit clears on a fixed ``jump_hold_time_s`` timer (0.85 s, the mean
//             training window). There is no foot-force sensing on this mujoco model,
//             so it cannot know whether the robot is still in the air.
//
// A jump that stays airborne past 0.85 s therefore had the check re-arm while still
// in flight -- which is precisely what tanaka reports: "the longer the airborne time,
// the more it ends up in a strange pose in the air". The window is now measured from
// the trigger alone and does not care whether the command bit has already timed out.
// ``max_jump_duration_s`` (2.2 s) still bounds everything from the far side.
bool jump_orientation_relaxed(ManagerBasedRLEnv* env)
{
    float relax_window_s = 1.0f;  // TerminationsCfgLongJump.bad_orientation.relax_window_s
    const auto cmd_cfg = env->cfg["commands"]["jump_command"];
    if (cmd_cfg.IsDefined() && cmd_cfg["orientation_relax_window_s"].IsDefined())
    {
        relax_window_s = cmd_cfg["orientation_relax_window_s"].as<float>();
    }

    return g_jump_since_trigger_wall >= 0.0f && g_jump_since_trigger_wall < relax_window_s;
}

// Instrumentation (2026-09-06): the first fix did not stop the freeze, and there is no
// measurement yet that says whether the FSM drops to Passive *in the air* or only after
// a genuine fall. Logging the tilt together with the jump clock at the instant the check
// fires separates the two: a Passive at ~0.9 s after the trigger with the robot still
// nose-high is a mid-flight cut-out, one seconds later is a real fall. Without this the
// next loop would be guesswork -- the failure mode
// プロジェクト_報酬項の名前と実測対象のズレ7例.md exists to prevent.
bool jump_bad_orientation_check(ManagerBasedRLEnv* env, float limit_angle)
{
    const bool relaxed = jump_orientation_relaxed(env);
    if (relaxed)
    {
        return false;
    }

    if (!isaaclab::mdp::bad_orientation(env, limit_angle))
    {
        return false;
    }

    const float tilt = std::fabs(std::acos(-env->robot->data.projected_gravity_b[2]));
    spdlog::info("bad_orientation -> Passive: tilt={:.3f} rad (limit {:.2f}), jump_cmd={:.0f}, "
                 "t_since_trigger={:.3f}s (bit clock {:.3f}s)",
                 tilt, limit_angle, g_jump_cmd, g_jump_since_trigger_wall, g_jump_time_since_trigger);
    return true;
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
            [&]()->bool{ return isaaclab::jump_bad_orientation_check(env.get(), 1.0); },
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
