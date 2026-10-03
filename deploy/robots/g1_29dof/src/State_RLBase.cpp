#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <cstdio>

namespace isaaclab
{
namespace {

// Q を押している間だけ走行モード。
// ターミナルのオートリピートは初回リピートまで ~500ms 空くので、
// 最後に 'q' を読んでから HOLD_MS の間はモードを保持してチャタリングを防ぐ。
constexpr int HOLD_MS = 600;

bool is_running(ManagerBasedRLEnv* env)
{
    static long cached_step = -1;
    static bool cached      = false;
    static std::chrono::steady_clock::time_point last_run_key{};

    if (env->episode_length == cached_step) return cached;   // 同一ステップ内は使い回す
    cached_step = env->episode_length;

    const std::string key = FSMState::keyboard ? FSMState::keyboard->key() : std::string("");
    const auto now = std::chrono::steady_clock::now();

    if (key == "q")        last_run_key = now;                                      // 走行キー
    else if (!key.empty()) last_run_key = std::chrono::steady_clock::time_point{};  // 他キーで即解除

    cached = (now - last_run_key) < std::chrono::milliseconds(HOLD_MS);
    return cached;
}

float rate_limit(float target, float & state, float max_rate, float dt)
{
    const float step = max_rate * dt;
    state += std::clamp(target - state, -step, step);
    return state;
}

} // anonymous namespace

REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    const std::string key = FSMState::keyboard ? FSMState::keyboard->key() : std::string("");
    const auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];

    const float walk_speed = params["walk_speed"].as<float>(1.0f);
    const float run_speed  = params["run_speed"].as<float>(2.2f);
    const float lat_speed  = params["lat_speed"].as<float>(0.5f);
    const float yaw_rate   = params["yaw_rate"].as<float>(1.0f);
    const float accel      = params["accel"].as<float>(2.0f);   // [m/s^2]
    const float yaw_accel  = params["yaw_accel"].as<float>(3.0f);

    static const std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", { 1.0f,  0.0f,  0.0f}},
        {"s", {-1.0f,  0.0f,  0.0f}},
        {"a", { 0.0f,  1.0f,  0.0f}},
        {"d", { 0.0f, -1.0f,  0.0f}},
        {"e", { 0.0f,  0.0f, -1.0f}},   // 旋回（左も要るなら {"z", {0,0,1}} を追加）
        {"z", { 0.0f,  0.0f,  1.0f}}, 
    };

    std::vector<float> target = {0.0f, 0.0f, 0.0f};
    if (is_running(env)) {
        target[0] = run_speed;                  // 走行は直進のみ
    } else {
        auto it = key_commands.find(key);
        if (it != key_commands.end()) {
            target[0] = it->second[0] * walk_speed;
            target[1] = it->second[1] * lat_speed;
            target[2] = it->second[2] * yaw_rate;
        }
    }

    // 学習時のコマンド範囲でクリップ（元コードは cfg を読むだけで使っていなかった）
    target[0] = std::clamp(target[0], ranges["lin_vel_x"][0].as<float>(), ranges["lin_vel_x"][1].as<float>());
    target[1] = std::clamp(target[1], ranges["lin_vel_y"][0].as<float>(), ranges["lin_vel_y"][1].as<float>());
    target[2] = std::clamp(target[2], ranges["ang_vel_z"][0].as<float>(), ranges["ang_vel_z"][1].as<float>());

    // 歩行↔走行の遷移はここのレートで決まる（急に 0→2.2 にすると転ぶ）
    static float vx = 0.0f, vy = 0.0f, wz = 0.0f;
    std::vector<float> cmd(3);
    cmd[0] = rate_limit(target[0], vx, accel,     env->step_dt);
    cmd[1] = rate_limit(target[1], vy, accel,     env->step_dt);
    cmd[2] = rate_limit(target[2], wz, yaw_accel, env->step_dt);
    //return cmd;
    // --- debug: 指令値の確認（不要になったら消す） ---
    static int dbg_counter = 0;
    if (++dbg_counter % 10 == 0) {   // 50Hz なので 0.2秒ごと
        printf("\r[cmd] key=%-4s run=%d | vx=%+.2f vy=%+.2f wz=%+.2f   ",
               key.empty() ? "-" : key.c_str(),
               static_cast<int>(is_running(env)),
               cmd[0], cmd[1], cmd[2]);
        fflush(stdout);
    }

    return cmd;
}

}

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