// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <boost/bimap.hpp>
#include <string>
#include <any>
#include <functional>
#include <utility>
#include <vector>

inline boost::bimap<int, std::string> FSMStringMap;

// A condition that, once true, hands control to `target_state`. `label` names the
// condition so CtrlFSM can report *which* one fired: several unrelated conditions
// (joystick, lowstate timeout, fall detection) all lead to Passive, and on hardware
// the difference between them is the whole diagnosis.
struct TransitionCheck
{
    std::function<bool()> triggered;
    int target_state;
    std::string label;

    TransitionCheck(std::function<bool()> fn, int target, std::string check_label)
    : triggered(std::move(fn)), target_state(target), label(std::move(check_label))
    {
    }

    // Accepts the older `emplace_back(std::make_pair(fn, id))` form still used by the
    // per-robot RL states, which register their checks without a label.
    template <typename Fn>
    TransitionCheck(std::pair<Fn, int> pair)
    : triggered(std::move(pair.first)), target_state(pair.second), label("unlabeled")
    {
    }
};

class BaseState
{
public:
    BaseState(int state, std::string state_string) : state_(state) 
    {
        FSMStringMap.insert({state, state_string});
    }

    virtual void enter() {}

    virtual void pre_run() {}
    virtual void run() {}
    virtual void post_run() {}

    virtual void exit() {}

    std::string getStateString() { return FSMStringMap.left.at(state_); }
    int getState() {return state_; }
    bool isState(int state) { return state_ == state; }
    std::vector<TransitionCheck> registered_checks;
private:
    int state_;
};

using FsmFactory = std::function<std::shared_ptr<BaseState>(int, std::string)>;
using FsmMap     = std::unordered_map<std::string, FsmFactory>;

inline FsmMap& getFsmMap() {
    static FsmMap fsmMap;
    return fsmMap;
}

#define REGISTER_FSM(Derived) \
    inline std::shared_ptr<BaseState> __factory_##Derived(int s, std::string ss) {      \
        return std::make_shared<Derived>(s, ss);                                        \
    }                                                                                   \
    inline struct __registrar_##Derived {                                               \
        __registrar_##Derived() {                                                       \
            getFsmMap()[#Derived] = __factory_##Derived;                                \
        }                                                                               \
    } __registrar_instance_##Derived;                                                   \
    
