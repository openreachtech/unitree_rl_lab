#pragma once

#include <heightmap_generator/msg/height_map.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace go2
{

inline constexpr float kHeightScanEmpty = 0.0f;
inline constexpr float kHeightScanFlatDefault = 0.0f;
inline constexpr const char* kHeightScanTopic = "/heightmap/data";

// Runtime-configurable heightmap grid, read from config.yaml's
// FSM.Velocity.height_scan block in HeightScanUpdater::init() (any field left unset
// keeps its default below). Defaults match Go2-HM-Belief/Noisy's training-time grid:
// HEIGHT_SCAN_RESOLUTION=0.05, HEIGHT_SCAN_SIZE=(1.4, 1.0) in velocity_env_cfg_lidar.py
// -> 29x21 = 609 raw cells, gathered down to 388 by excluding the 0.40x0.30 body
// footprint (GO2_LIDAR_BODY_HALF_EXTENT_X/Y). To deploy a policy trained on a
// different grid instead (e.g. the older 17x11=187-cell top-down grid, which used the
// raster directly as the policy input with no exclusion gather), override these in
// config.yaml and set gather_enabled: false - no recompile needed.
struct HeightScanGridConfig
{
    int grid_nx = 29;
    int grid_ny = 21;
    float resolution = 0.05f;
    float x_min = -0.7f;
    float y_min = -0.5f;
    // Body-footprint half-extents excluded when gather_enabled is true.
    float exclude_half_x = 0.40f;
    float exclude_half_y = 0.30f;
    // true: gather the raster down to the cells outside the body-footprint exclusion
    // (what Go2-HM-Belief/Noisy were trained on). false: the raster IS the policy
    // input verbatim, no cells excluded (matches older policies whose HeightMap
    // message size already equalled their height_scan input size).
    bool gather_enabled = true;
};

// Consumes the raster heightmap_generator publishes (msg.width x msg.height cells,
// described by the message itself) and gathers it down to whatever cells the
// deployed policy actually expects, per HeightScanGridConfig. Configuring this at
// runtime (instead of hardcoding one grid) lets the same go2_ctrl binary serve either
// the LiDAR/Belief/Noisy 609->388 policies or an older/different-grid policy, chosen
// purely via config.yaml.
class HeightScanUpdater
{
public:
    static HeightScanUpdater& instance();

    void init();

    // Policy observation: the gathered cells, in keep-index order. Valid after init().
    std::vector<float> get() const;

    // Length of get()'s result (and what the deployed policy's height_scan input must
    // be sized for). Valid after init().
    std::size_t size() const { return keep_index_.size(); }

    std::vector<float> make_flat(float value = kHeightScanFlatDefault) const
    {
        return std::vector<float>(keep_index_.size(), value);
    }

private:
    HeightScanUpdater() = default;

    void on_height_scan(const heightmap_generator::msg::HeightMap::SharedPtr msg);
    bool parse_height_scan(
        const heightmap_generator::msg::HeightMap& msg,
        std::vector<float>& out) const;

    HeightScanGridConfig cfg_;
    // Flat raster indices (index = ix * cfg_.grid_ny + iy, matching heightmap_generator's
    // own "yx" ordering) of the cells kept, in the same order the training-time
    // observation term selects them in - get it out of step and the policy is fed real
    // height data through the wrong input slots with no error, just worse walking.
    // Identity (0..grid_nx*grid_ny-1) when cfg_.gather_enabled is false. Computed once
    // in init() from cfg_.
    std::vector<int> keep_index_;

    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<heightmap_generator::msg::HeightMap>::SharedPtr height_scan_sub_;
    rclcpp::executors::SingleThreadedExecutor executor_;
    std::thread spin_thread_;

    mutable std::mutex mutex_;
    std::vector<float> height_scan_;
};

} // namespace go2
