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

// Isaac Lab GridPatternCfg(resolution=0.1, size=[1.6, 1.0]) → 17×11 = 187
inline constexpr int kHeightScanGridNx = 17;
inline constexpr int kHeightScanGridNy = 11;
inline constexpr int kHeightScanSize = kHeightScanGridNx * kHeightScanGridNy;
inline constexpr float kHeightScanResolution = 0.1f;
inline constexpr float kHeightScanXMin = -0.6f;
inline constexpr float kHeightScanYMin = -0.5f;
inline constexpr float kHeightScanEmpty = 0.0f;
inline constexpr const char* kHeightScanTopic = "/heightmap/data";

inline std::vector<float> make_default_height_scan()
{
    return std::vector<float>(kHeightScanSize, kHeightScanEmpty);
}

inline constexpr float kHeightScanFlatDefault = 0.0f;

inline std::vector<float> make_flat_height_scan(float value = kHeightScanFlatDefault)
{
    return std::vector<float>(kHeightScanSize, value);
}

// Consumes the final x-major 17x11 map published by heightmap_generator.
class HeightScanUpdater
{
public:
    static HeightScanUpdater& instance();

    void init();

    // Policy observation in heightmap_generator order: index = ix * 11 + iy.
    std::vector<float> get() const;

private:
    HeightScanUpdater() = default;

    void on_height_scan(const heightmap_generator::msg::HeightMap::SharedPtr msg);
    static bool parse_height_scan(
        const heightmap_generator::msg::HeightMap& msg,
        std::vector<float>& out);

    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<heightmap_generator::msg::HeightMap>::SharedPtr height_scan_sub_;
    rclcpp::executors::SingleThreadedExecutor executor_;
    std::thread spin_thread_;

    mutable std::mutex mutex_;
    std::vector<float> height_scan_ = make_default_height_scan();
};

} // namespace go2
