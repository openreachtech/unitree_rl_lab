#pragma once

#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/ros2/PointCloud2_.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace go2
{

// Isaac Lab GridPatternCfg(resolution=0.05, size=[1.4, 1.0]) → 29×21 = 609 cells, grid
// centered on the body origin (not the LiDAR mount).
//
// All 609 reach the policy. The 13×9 cells under the body used to be dropped, because the
// scanner that fed this was mounted on top of the trunk and could not see beneath it. The
// L1 sits at the nose looking down and back -- its steepest rays land 0.15 m from the
// mount, inside the footprint -- so those cells carry real measurements now, and the
// perceptive policies are trained on the full grid (exclude_half_extent < 0 in
// velocity_env_cfg_mid360.py).
inline constexpr int kHeightScanGridNx = 29;
inline constexpr int kHeightScanGridNy = 21;
inline constexpr int kHeightScanSize = kHeightScanGridNx * kHeightScanGridNy;
// Matches velocity_env_cfg_go2.GO2_HEIGHT_SCAN_OFFSET
// (= GO2_NOMINAL_BASE_Z + GO2_LIDAR_OFFSET_Z = 0.32 - 0.046825).
inline constexpr float kHeightScanOffset = 0.273175f;
inline constexpr float kHeightScanClipMin = -1.0f;
inline constexpr float kHeightScanClipMax = 5.0f;
inline constexpr const char* kHeightScanTopic = "rt/height_scan";


// What a cell reads on flat ground at nominal stance: GO2_NOMINAL_BASE_Z minus
// kHeightScanOffset. This is velocity_env_cfg_go2.GO2_FLAT_SCAN_VALUE, and it is what a
// LiDAR-built map holds in a cell no beam has reached yet -- including before the first
// message arrives.
inline constexpr float kHeightScanFlatDefault = 0.046825f;

inline std::vector<float> make_default_height_scan()
{
    return std::vector<float>(kHeightScanSize, kHeightScanFlatDefault);
}

inline std::vector<float> make_flat_height_scan(float value = kHeightScanFlatDefault)
{
    return std::vector<float>(kHeightScanSize, value);
}

// Consumes a ready-made height map (e.g. MuJoCo HeightMapSimulator on rt/height_scan),
// assumed to be sampled on a grid centered at the base body origin (see
// kGridCenterOffsetX/Y) and to use the same height convention as training:
// base_z - hit_z - kHeightScanOffset.
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

    std::shared_ptr<unitree::robot::SubscriptionBase<sensor_msgs::msg::dds_::PointCloud2_>>
        height_scan_sub_;

    mutable std::mutex mutex_;
    std::vector<float> height_scan_ = make_default_height_scan();
    std::uint64_t msg_count_ = 0;
};

} // namespace go2
