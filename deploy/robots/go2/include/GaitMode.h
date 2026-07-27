#pragma once

#include <atomic>
#include <string>
#include <vector>

namespace go2
{

// Stance commanded to the multimode policy (Isaac Lab task `Unitree-Go2-Multimode`).
// The ids and the one-hot layout below are what the policy was trained on, so they
// must stay in sync with MODE_QUAD / MODE_HIND_BIPED / MODE_FRONT_BIPED in
// unitree_rl_lab.tasks.biped.mdp.commands.
enum class GaitMode : int
{
    kQuad = 0,
    kHindBiped = 1,
    kFrontBiped = 2,
};

inline constexpr int kGaitModeCount = 3;

inline const char* gait_mode_name(GaitMode mode)
{
    switch (mode)
    {
    case GaitMode::kQuad: return "quad";
    case GaitMode::kHindBiped: return "hind_biped";
    case GaitMode::kFrontBiped: return "front_biped";
    }
    return "unknown";
}

inline bool parse_gait_mode(const std::string& name, GaitMode& mode)
{
    for (int i = 0; i < kGaitModeCount; ++i)
    {
        const auto candidate = static_cast<GaitMode>(i);
        if (name == gait_mode_name(candidate))
        {
            mode = candidate;
            return true;
        }
    }
    return false;
}

// Written by the FSM thread from operator input, read by the `gait_mode` observation
// on the policy thread.
class GaitModeSelector
{
public:
    static GaitModeSelector& instance()
    {
        static GaitModeSelector selector;
        return selector;
    }

    void set(GaitMode mode) { mode_.store(static_cast<int>(mode), std::memory_order_relaxed); }

    GaitMode get() const { return static_cast<GaitMode>(mode_.load(std::memory_order_relaxed)); }

    std::vector<float> one_hot() const
    {
        std::vector<float> obs(kGaitModeCount, 0.0f);
        obs[static_cast<int>(get())] = 1.0f;
        return obs;
    }

private:
    GaitModeSelector() = default;

    std::atomic<int> mode_{static_cast<int>(GaitMode::kQuad)};
};

} // namespace go2
