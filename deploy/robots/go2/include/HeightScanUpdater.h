#pragma once

#include "Types.h"

#include <unitree/dds_wrapper/common/Publisher.h>
#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/idl/ros2/PointCloud2_.hpp>

#include <Eigen/Dense>
#include <cstdint>
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
inline constexpr float kHeightScanOffset = 0.5f;
inline constexpr float kHeightScanClipMin = -1.0f;
inline constexpr float kHeightScanClipMax = 5.0f;
inline constexpr float kHeightScanEmpty = -1.0f;
inline constexpr const char* kHeightScanTopic = "rt/height_scan";

inline std::vector<float> make_default_height_scan()
{
    return std::vector<float>(kHeightScanSize, kHeightScanEmpty);
}

class HeightScanUpdater
{
public:
    static HeightScanUpdater& instance();

    void init(const std::shared_ptr<LowState_t>& lowstate);

    std::vector<float> get() const;

private:
    HeightScanUpdater() = default;

    void on_cloud(const sensor_msgs::msg::dds_::PointCloud2_& cloud);
    void init_publisher();
    void publish_height_scan(
        const std::vector<float>& scan,
        const sensor_msgs::msg::dds_::PointCloud2_& cloud);
    std::vector<float> compute_height_scan(
        const sensor_msgs::msg::dds_::PointCloud2_& cloud,
        const Eigen::Quaternionf& imu_quat) const;

    static bool parse_xyz_offsets(
        const sensor_msgs::msg::dds_::PointCloud2_& cloud,
        int& x_offset,
        int& y_offset,
        int& z_offset);

    static Eigen::Matrix3f mount_correction_matrix();
    static Eigen::Vector3f yaw_level_point(
        const Eigen::Vector3f& p_body,
        const Eigen::Quaternionf& imu_quat);

    static float grid_index_to_height(float min_z);
    static void median_fill(std::vector<float>& grid);
    static float clip_height(float value);

    std::shared_ptr<LowState_t> lowstate_;
    std::shared_ptr<unitree::robot::SubscriptionBase<sensor_msgs::msg::dds_::PointCloud2_>> cloud_sub_;
    unitree::robot::RealTimePublisher<sensor_msgs::msg::dds_::PointCloud2_> height_scan_pub_{
        kHeightScanTopic};
    bool publisher_ready_ = false;

    mutable std::mutex mutex_;
    std::vector<float> height_scan_ = make_default_height_scan();
};

} // namespace go2
