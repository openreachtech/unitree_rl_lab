#pragma once

#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/ros2/PointCloud2_.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace go2
{

// Isaac Lab GridPatternCfg(resolution=0.1, size=[1.6, 1.0]) → 17×11 = 187
inline constexpr int kHeightScanGridNx = 17;
inline constexpr int kHeightScanGridNy = 11;
inline constexpr int kHeightScanSize = kHeightScanGridNx * kHeightScanGridNy;
inline constexpr float kHeightScanSizeX = 1.6f;
inline constexpr float kHeightScanSizeY = 1.0f;
inline constexpr float kHeightScanResolution = 0.1f;
// Matches velocity_env_cfg_go2.GO2_HEIGHT_SCAN_OFFSET
// (= GO2_NOMINAL_BASE_Z + GO2_LIDAR_OFFSET_Z = 0.32 - 0.046825).
inline constexpr float kHeightScanOffset = 0.273175f;
inline constexpr float kHeightScanClipMin = -1.0f;
inline constexpr float kHeightScanClipMax = 5.0f;
inline constexpr float kHeightScanEmpty = -1.0f;
inline constexpr const char* kHeightScanTopic = "rt/height_scan_processed";

// Matches POLICY_HEIGHT_SCAN_CFG / height_scan_excluding_body.
inline constexpr float kLidarOffsetX = 0.28945f;
inline constexpr float kLidarOffsetY = 0.0f;
inline constexpr float kExcludeHalfExtentX = 0.22f;
inline constexpr float kExcludeHalfExtentY = 0.12f;
inline constexpr float kExcludeFillValue = 0.0f;

inline std::vector<float> make_default_height_scan()
{
    return std::vector<float>(kHeightScanSize, kHeightScanEmpty);
}

inline constexpr float kHeightScanFlatDefault = 0.0f;

inline std::vector<float> make_flat_height_scan(float value = kHeightScanFlatDefault)
{
    return std::vector<float>(kHeightScanSize, value);
}

// Consumes a ready-made height map (e.g. MuJoCo HeightMapSimulator on rt/height_scan).
// get() applies under-body masking to match training height_scan_excluding_body.
class HeightScanUpdater
{
public:
    static HeightScanUpdater& instance();

    void init();

    // Policy observation: raw map with under-body cells filled.
    std::vector<float> get() const;

private:
    HeightScanUpdater() = default;

    void on_height_scan(const sensor_msgs::msg::dds_::PointCloud2_& msg);
    static bool parse_height_scan(
        const sensor_msgs::msg::dds_::PointCloud2_& msg,
        std::vector<float>& out);
    static std::vector<float> exclude_under_body(const std::vector<float>& scan);

    std::shared_ptr<unitree::robot::SubscriptionBase<sensor_msgs::msg::dds_::PointCloud2_>>
        height_scan_sub_;

    mutable std::mutex mutex_;
    std::vector<float> height_scan_ = make_default_height_scan();
};

} // namespace go2
