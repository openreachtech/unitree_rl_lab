#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/utils/step_recorder.h"
#include <condition_variable>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <spdlog/spdlog.h>
#include <sstream>
#include <thread>

/// One CSV per entry into the state, one row per policy step, for sim-to-real comparison
/// (the same logger runs against unitree_mujoco and the robot).
///
/// Columns: t, cmd (3), obs (policy input), action (policy output, unscaled),
/// then per motor in SDK order: q_target, q, dq, tau (tau_est), temp,
/// then IMU quat (w x y z), gyro, acc.
/// obs / action are in the policy's (Isaac Lab) joint order; cmd is obs[6:9].
/// Rows are formatted in the policy thread and written by a separate thread.
class CsvRecorder : public isaaclab::StepRecorder
{
public:
    explicit CsvRecorder(std::filesystem::path dir) : dir_(std::move(dir)) {}
    ~CsvRecorder() override { end(); }

    void begin(const isaaclab::ManagerBasedRLEnv& env) override
    {
        end();
        char stamp[32];
        std::time_t now = std::time(nullptr);
        std::strftime(stamp, sizeof(stamp), "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
        path_ = dir_ / (std::string(stamp) + ".csv");
        std::error_code ec;
        std::filesystem::create_directories(dir_, ec);
        file_.open(path_);
        if (!file_) {  // never stop the controller over a log file
            spdlog::warn("Cannot write {} ({}); not logging this run", path_.string(), ec.message());
            return;
        }
        header_written_ = false;
        running_ = true;
        writer_ = std::thread([this] { write_loop(); });
        spdlog::info("Logging to {}", path_.string());
    }

    void record(const isaaclab::ManagerBasedRLEnv& env, double t) override
    {
        if (!writer_.joinable()) return;  // begin() failed
        std::vector<float> obs;
        for (const auto& [name, v] : std::map(env.last_obs.begin(), env.last_obs.end()))  // stable order
            obs.insert(obs.end(), v.begin(), v.end());
        const auto& map = env.robot->data.joint_ids_map;
        const auto processed = env.action_manager->processed_actions();
        std::vector<float> q_target(map.size());
        for (size_t i = 0; i < map.size(); ++i) q_target[int(map[i])] = processed[i];

        std::ostringstream row;
        if (!header_written_) {
            row << header(obs.size(), env.last_action.size(), map.size()) << '\n';
            header_written_ = true;
        }
        row << t;
        for (size_t i = 6; i < 9 && i < obs.size(); ++i) row << ',' << obs[i];
        for (float v : obs) row << ',' << v;
        for (float v : env.last_action) row << ',' << v;
        for (float v : q_target) row << ',' << v;
        {
            const auto& ls = FSMState::lowstate;
            std::lock_guard<std::mutex> lock(ls->mutex_);
            const auto& m = ls->msg_.motor_state();
            const size_t n = map.size();
            for (size_t i = 0; i < n; ++i) row << ',' << m[i].q();
            for (size_t i = 0; i < n; ++i) row << ',' << m[i].dq();
            for (size_t i = 0; i < n; ++i) row << ',' << m[i].tau_est();
            for (size_t i = 0; i < n; ++i) row << ',' << m[i].temperature()[0];
            const auto& imu = ls->msg_.imu_state();
            for (float v : imu.quaternion()) row << ',' << v;
            for (float v : imu.gyroscope()) row << ',' << v;
            for (float v : imu.accelerometer()) row << ',' << v;
        }
        row << '\n';
        {
            std::lock_guard<std::mutex> lock(mtx_);
            queue_.push_back(row.str());
        }
        cv_.notify_one();
    }

    void end() override
    {
        if (!writer_.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            running_ = false;
        }
        cv_.notify_one();
        writer_.join();
        file_.close();
        spdlog::info("Log saved: {}", path_.string());
    }

private:
    static std::string header(size_t n_obs, size_t n_act, size_t n_joint)
    {
        std::ostringstream h;
        h << "t,cmd_vx,cmd_vy,cmd_wz";
        for (size_t i = 0; i < n_obs; ++i) h << ",obs_" << i;
        for (size_t i = 0; i < n_act; ++i) h << ",action_" << i;
        for (const char* name : {"q_target", "q", "dq", "tau", "temp"})
            for (size_t i = 0; i < n_joint; ++i) h << ',' << name << '_' << i;
        h << ",quat_w,quat_x,quat_y,quat_z,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z";
        return h.str();
    }

    void write_loop()
    {
        std::unique_lock<std::mutex> lock(mtx_);
        while (true) {
            cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
            std::deque<std::string> rows;
            rows.swap(queue_);
            const bool stop = !running_;
            lock.unlock();
            for (const auto& r : rows) file_ << r;
            file_.flush();
            lock.lock();
            if (stop && queue_.empty()) return;
        }
    }

    std::filesystem::path dir_, path_;
    std::ofstream file_;
    bool header_written_ = false;
    std::thread writer_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<std::string> queue_;
    bool running_ = false;
};
