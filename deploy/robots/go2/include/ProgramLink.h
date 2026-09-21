// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

// ProgramLink -- the one port the language side has into this controller.
//
// Everything above the velocity command (the skill grammar, the compiler that turns a program into
// a 50 Hz command stream, the state block the model reads, the conversation) lives in Python, in
// `unitree_rl_lab.program` and `scripts/llm/`. That code is what the dataset was built with and what
// the model was trained against, so a second implementation here would be a second thing to keep in
// sync -- and the moment it drifted, the model would be reading a state block it was never trained
// on. See scripts/llm/SYSTEM.md.
//
// So this class is deliberately thin: a UDP receive thread that parks a velocity target and a
// couple of queued triggers where State_Multitask can pick them up on its own step boundary, plus a
// socket to publish state back on. No policy, no FSM, no timing of its own.
//
// Wire format, one command per line, UTF-8, datagrams may carry several lines:
//
//   VEL <vx> <vy> <wz>      body-frame target, m/s and rad/s. Also the freshness heartbeat.
//   FLIP <kind>             backflip | frontflip | sideflip_left | sideflip_right | jump
//   STANCE front|hind|off   bipedal stance on (front/hind legs) or down
//   STOP                    zero the target now and drop anything queued
//   RESUME                  clear the manual-stop latch (see below)
//   PING                    no-op; the reply is the state stream itself
//
// Freshness: a velocity target counts only for `timeout_s` after the datagram that carried it. The
// conductor sends at 50 Hz, so 0.5 s is 25 missed packets -- a crashed conductor hands the robot
// back to the keyboard rather than leaving it running at the last commanded speed.
//
// Manual-stop latch: [Space] on the keyboard is the operator's safety valve, and it has to beat the
// network. Pressing it latches the link off -- `velocity()` reports nothing fresh, so the keyboard
// (whose own command the space bar just zeroed) is back in charge -- until the conductor sends
// RESUME. The latch is reported in the state line so the conductor knows to drop its queue instead
// of quietly fighting the operator.

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

class ProgramLink
{
public:
    struct Config
    {
        bool enabled = false;
        int port = 7777;               // where commands arrive
        int state_port = 7778;         // where the state line is sent
        std::string state_host = "127.0.0.1";
        float timeout_s = 0.5f;
    };

    explicit ProgramLink(const Config & cfg);
    ~ProgramLink();

    ProgramLink(const ProgramLink &) = delete;
    ProgramLink & operator=(const ProgramLink &) = delete;

    // Bind and start the receive thread. Idempotent; logs and gives up if the port is taken.
    void start();
    void stop();
    bool running() const { return running_.load(); }

    // The current network target, if one arrived within timeout_s and the manual-stop latch is
    // clear. False means "not driving" -- the caller should fall back to its own source.
    bool velocity(std::array<float, 3> & out) const;

    // Queued triggers, taken one per call. Queued rather than applied because the command clocks
    // they feed advance on the policy thread and must begin on a step boundary -- the same reason
    // the keyboard path in State_Multitask queues.
    bool take_flip(std::string & kind);
    bool take_stance(float & sign);   // +1 front stance, -1 hind, 0 come down

    void note_manual_stop();
    bool manual_stopped() const { return manual_stop_.load(); }

    // One line of state, sent to state_host:state_port. Fire and forget; a missing listener is
    // not an error here.
    void publish(const std::string & line);

    // Drop the target and everything queued, and clear the latch. Called when the state is entered.
    void reset();

    // The link the velocity observation term should read, set while State_Multitask holds one.
    // A raw pointer rather than a shared_ptr: the term runs on the policy thread every step, and
    // the owning state outlives it (exit() joins that thread before clearing this).
    static std::atomic<ProgramLink *> active;

private:
    void receive_loop();
    void handle_line(const std::string & line);

    Config cfg_;
    int rx_fd_ = -1;
    int tx_fd_ = -1;

    std::atomic<bool> running_{false};
    std::thread thread_;

    std::atomic<float> vx_{0.0f};
    std::atomic<float> vy_{0.0f};
    std::atomic<float> wz_{0.0f};
    // steady_clock nanoseconds of the last VEL/STOP; negative before the first one.
    std::atomic<long long> last_rx_ns_{-1};
    std::atomic<bool> manual_stop_{false};

    mutable std::mutex queue_mutex_;
    std::deque<std::string> flips_;
    std::deque<float> stances_;

    std::atomic<long long> datagrams_{0};
};
