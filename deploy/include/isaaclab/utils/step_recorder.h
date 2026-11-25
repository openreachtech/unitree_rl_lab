#pragma once

namespace isaaclab
{
class ManagerBasedRLEnv;

/// Called by State_RLBase around its policy loop: begin() when the state is entered,
/// record() after every policy step (in the policy thread), end() on exit.
class StepRecorder
{
public:
    virtual ~StepRecorder() = default;
    virtual void begin(const ManagerBasedRLEnv& env) = 0;
    virtual void record(const ManagerBasedRLEnv& env, double t) = 0;
    virtual void end() = 0;
};
}  // namespace isaaclab
