#include "HeightScanUpdater.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <spdlog/spdlog.h>

namespace go2
{

namespace
{

constexpr float kMountRoll = -2.9130f;
constexpr float kMountPitch = -0.1320f;
constexpr float kMountYaw = -1.0570f;

constexpr float kHeightFilterMax = 0.4f;
constexpr float kHeightFilterMin = -0.3f;

constexpr float kBodyFilterXMin = -0.7f;
constexpr float kBodyFilterXMax = -0.1f;
constexpr float kBodyFilterYMin = -0.3f;
constexpr float kBodyFilterYMax = 0.3f;
constexpr float kBodyFilterZMin = -0.3f;
constexpr float kBodyFilterZMax = 0.05f;

constexpr uint8_t kPointFieldFloat32 = 7;

bool is_finite(float value)
{
    return std::isfinite(value);
}

} // namespace

HeightScanUpdater& HeightScanUpdater::instance()
{
    static HeightScanUpdater updater;
    return updater;
}

void HeightScanUpdater::init(const std::shared_ptr<LowState_t>& lowstate)
{
    lowstate_ = lowstate;
    cloud_sub_ = std::make_shared<
        unitree::robot::SubscriptionBase<sensor_msgs::msg::dds_::PointCloud2_>>(
        "rt/utlidar/cloud",
        [this](const void* msg) {
            on_cloud(*static_cast<const sensor_msgs::msg::dds_::PointCloud2_*>(msg));
        });
    cloud_sub_->set_timeout_ms(500);
    spdlog::info("HeightScanUpdater subscribed to rt/utlidar/cloud");
}

std::vector<float> HeightScanUpdater::get() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return height_scan_;
}

void HeightScanUpdater::on_cloud(const sensor_msgs::msg::dds_::PointCloud2_& cloud)
{
    if (!lowstate_)
    {
        return;
    }

    Eigen::Quaternionf imu_quat;
    {
        std::lock_guard<std::mutex> lock(lowstate_->mutex_);
        const auto& q = lowstate_->msg_.imu_state().quaternion();
        imu_quat = Eigen::Quaternionf(q[0], q[1], q[2], q[3]);
    }

    const auto scan = compute_height_scan(cloud, imu_quat);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        height_scan_ = scan;
    }
}

bool HeightScanUpdater::parse_xyz_offsets(
    const sensor_msgs::msg::dds_::PointCloud2_& cloud,
    int& x_offset,
    int& y_offset,
    int& z_offset)
{
    x_offset = -1;
    y_offset = -1;
    z_offset = -1;

    for (const auto& field : cloud.fields())
    {
        if (field.datatype() != kPointFieldFloat32)
        {
            continue;
        }
        if (field.name() == "x")
        {
            x_offset = static_cast<int>(field.offset());
        }
        else if (field.name() == "y")
        {
            y_offset = static_cast<int>(field.offset());
        }
        else if (field.name() == "z")
        {
            z_offset = static_cast<int>(field.offset());
        }
    }

    return x_offset >= 0 && y_offset >= 0 && z_offset >= 0;
}

Eigen::Matrix3f HeightScanUpdater::mount_correction_matrix()
{
    const Eigen::Matrix3f rotation =
        (Eigen::AngleAxisf(kMountYaw, Eigen::Vector3f::UnitZ())
         * Eigen::AngleAxisf(kMountPitch, Eigen::Vector3f::UnitY())
         * Eigen::AngleAxisf(kMountRoll, Eigen::Vector3f::UnitX()))
            .toRotationMatrix();
    return rotation;
}

Eigen::Vector3f HeightScanUpdater::yaw_level_point(
    const Eigen::Vector3f& p_body,
    const Eigen::Quaternionf& imu_quat)
{
    const Eigen::Matrix3f R_body = imu_quat.normalized().toRotationMatrix();
    const Eigen::Vector3f p_world = R_body * p_body;

    const float yaw = std::atan2(R_body(1, 0), R_body(0, 0));
    const float cy = std::cos(yaw);
    const float sy = std::sin(yaw);

    Eigen::Vector3f p_level;
    p_level.x() = cy * p_world.x() + sy * p_world.y();
    p_level.y() = -sy * p_world.x() + cy * p_world.y();
    p_level.z() = p_world.z();
    return p_level;
}

float HeightScanUpdater::grid_index_to_height(float min_z)
{
    // Isaac Lab: pos_w.z - hit_z - offset, with sensor origin at z=0.
    return -min_z - kHeightScanOffset;
}

void HeightScanUpdater::median_fill(std::vector<float>& grid)
{
    const auto original = grid;
    const int nx = kHeightScanGridNx;
    const int ny = kHeightScanGridNy;

    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            const int idx = ix * ny + iy;
            if (original[idx] > kHeightScanEmpty + 1e-3f)
            {
                continue;
            }

            std::vector<float> neighbors;
            neighbors.reserve(9);
            for (int dx = -1; dx <= 1; ++dx)
            {
                for (int dy = -1; dy <= 1; ++dy)
                {
                    const int nx_idx = ix + dx;
                    const int ny_idx = iy + dy;
                    if (nx_idx < 0 || nx_idx >= nx || ny_idx < 0 || ny_idx >= ny)
                    {
                        continue;
                    }
                    const float value = original[nx_idx * ny + ny_idx];
                    if (value > kHeightScanEmpty + 1e-3f)
                    {
                        neighbors.push_back(value);
                    }
                }
            }

            if (!neighbors.empty())
            {
                std::nth_element(
                    neighbors.begin(),
                    neighbors.begin() + neighbors.size() / 2,
                    neighbors.end());
                grid[idx] = neighbors[neighbors.size() / 2];
            }
        }
    }
}

float HeightScanUpdater::clip_height(float value)
{
    return std::clamp(value, kHeightScanClipMin, kHeightScanClipMax);
}

std::vector<float> HeightScanUpdater::compute_height_scan(
    const sensor_msgs::msg::dds_::PointCloud2_& cloud,
    const Eigen::Quaternionf& imu_quat) const
{
    int x_offset = 0;
    int y_offset = 0;
    int z_offset = 0;
    if (!parse_xyz_offsets(cloud, x_offset, y_offset, z_offset))
    {
        spdlog::warn("HeightScanUpdater: PointCloud2 missing x/y/z float fields");
        return make_default_height_scan();
    }

    const auto& data = cloud.data();
    if (cloud.point_step() == 0 || data.empty())
    {
        return make_default_height_scan();
    }

    const uint32_t num_points = cloud.width() * cloud.height();
    const uint32_t point_step = cloud.point_step();
    const Eigen::Matrix3f R_mount = mount_correction_matrix();

    const float half_x = 0.5f * kHeightScanSizeX;
    const float half_y = 0.5f * kHeightScanSizeY;
    const float x_min = -half_x;
    const float y_min = -half_y;

    std::vector<float> grid(kHeightScanSize, kHeightScanEmpty);
    std::vector<float> min_z(kHeightScanSize, std::numeric_limits<float>::infinity());

    for (uint32_t point_idx = 0; point_idx < num_points; ++point_idx)
    {
        const uint32_t base = point_idx * point_step;
        if (base + point_step > data.size())
        {
            break;
        }

        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        std::memcpy(&x, data.data() + base + x_offset, sizeof(float));
        std::memcpy(&y, data.data() + base + y_offset, sizeof(float));
        std::memcpy(&z, data.data() + base + z_offset, sizeof(float));

        if (!is_finite(x) || !is_finite(y) || !is_finite(z))
        {
            continue;
        }

        // [2] mount correction (LiDAR → body-forward frame)
        const Eigen::Vector3f p_body = R_mount * Eigen::Vector3f(x, y, z);

        // [3] IMU yaw leveling (roll/pitch removed, world-vertical z)
        const Eigen::Vector3f p = yaw_level_point(p_body, imu_quat);

        // [4] height filter
        if (p.z() > kHeightFilterMax || p.z() < kHeightFilterMin)
        {
            continue;
        }

        // [5] body filter box
        if (p.x() > kBodyFilterXMin && p.x() < kBodyFilterXMax
            && p.y() > kBodyFilterYMin && p.y() < kBodyFilterYMax
            && p.z() > kBodyFilterZMin && p.z() < kBodyFilterZMax)
        {
            continue;
        }

        // [6] grid binning
        if (p.x() < x_min || p.x() > half_x || p.y() < y_min || p.y() > half_y)
        {
            continue;
        }

        const int ix = static_cast<int>(std::floor((p.x() - x_min) / kHeightScanResolution));
        const int iy = static_cast<int>(std::floor((p.y() - y_min) / kHeightScanResolution));
        if (ix < 0 || ix >= kHeightScanGridNx || iy < 0 || iy >= kHeightScanGridNy)
        {
            continue;
        }

        const int cell = ix * kHeightScanGridNy + iy;
        min_z[cell] = std::min(min_z[cell], p.z());
    }

    for (int cell = 0; cell < kHeightScanSize; ++cell)
    {
        if (std::isfinite(min_z[cell]))
        {
            grid[cell] = grid_index_to_height(min_z[cell]);
        }
    }

    // [7] median fill (optional in doc, enabled by default)
    median_fill(grid);

    // [8] clip
    for (float& value : grid)
    {
        if (value > kHeightScanEmpty + 1e-3f)
        {
            value = clip_height(value);
        }
    }

    return grid;
}

} // namespace go2
