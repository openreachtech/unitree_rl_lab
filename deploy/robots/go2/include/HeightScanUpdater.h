#pragma once

#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/ros2/PointCloud2_.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace go2
{

// Isaac Lab GridPatternCfg(resolution=0.05, size=[1.4, 1.0]) → 29×21 = 609 raw cells, grid
// centered on the body origin (not the LiDAR mount). The 13×9 cells under the body are
// removed before policy inference.
inline constexpr int kHeightScanGridNx = 29;
inline constexpr int kHeightScanGridNy = 21;
inline constexpr int kHeightScanRawSize = kHeightScanGridNx * kHeightScanGridNy;
inline constexpr int kHeightScanSize = kHeightScanRawSize - 13 * 9;
inline constexpr float kHeightScanSizeX = 1.4f;
inline constexpr float kHeightScanSizeY = 1.0f;
inline constexpr float kHeightScanResolution = 0.05f;
// Matches velocity_env_cfg_go2.GO2_HEIGHT_SCAN_OFFSET
// (= GO2_NOMINAL_BASE_Z + GO2_LIDAR_OFFSET_Z = 0.32 - 0.046825).
inline constexpr float kHeightScanOffset = 0.273175f;
inline constexpr float kHeightScanClipMin = -1.0f;
inline constexpr float kHeightScanClipMax = 5.0f;
inline constexpr float kHeightScanEmpty = -1.0f;
inline constexpr const char* kHeightScanTopic = "rt/height_scan";

// Matches POLICY_HEIGHT_SCAN_CFG / height_scan_excluding_body. Grid center offset from base
// origin -- 0 because the grid is body-centered, not LiDAR-mount-centered (a LiDAR-centered
// grid barely reached behind the robot).
inline constexpr float kGridCenterOffsetX = 0.0f;
inline constexpr float kGridCenterOffsetY = 0.0f;
inline constexpr float kExcludeHalfExtentX = 0.30f;
inline constexpr float kExcludeHalfExtentY = 0.20f;

inline std::vector<float> make_default_height_scan()
{
    return std::vector<float>(kHeightScanSize, kHeightScanEmpty);
}

inline std::vector<float> make_default_raw_height_scan()
{
    return std::vector<float>(kHeightScanRawSize, kHeightScanEmpty);
}

inline constexpr float kHeightScanFlatDefault = 0.0f;

inline std::vector<float> make_flat_height_scan(float value = kHeightScanFlatDefault)
{
    return std::vector<float>(kHeightScanSize, value);
}

// Consumes a ready-made height map (e.g. MuJoCo HeightMapSimulator on rt/height_scan), assumed
// to be sampled on a grid centered at the base body origin (see kGridCenterOffsetX/Y).
// get() removes under-body cells to match training height_scan_excluding_body.
class HeightScanUpdater
{
public:
    static HeightScanUpdater& instance();

    void init();

    // Policy observation with under-body cells removed.
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
    std::vector<float> height_scan_ = make_default_raw_height_scan();
};

} // namespace go2
