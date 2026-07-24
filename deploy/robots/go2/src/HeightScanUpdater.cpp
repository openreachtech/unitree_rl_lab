#include "HeightScanUpdater.h"

#include <cmath>
#include <cstring>
#include <spdlog/spdlog.h>

namespace go2
{

HeightScanUpdater& HeightScanUpdater::instance()
{
    static HeightScanUpdater updater;
    return updater;
}

void HeightScanUpdater::init()
{
    height_scan_sub_ = std::make_shared<
        unitree::robot::SubscriptionBase<sensor_msgs::msg::dds_::PointCloud2_>>(
        kHeightScanTopic,
        [this](const void* msg) {
            on_height_scan(*static_cast<const sensor_msgs::msg::dds_::PointCloud2_*>(msg));
        });
    height_scan_sub_->set_timeout_ms(500);
    spdlog::info("HeightScanUpdater subscribed to {} (direct height map)", kHeightScanTopic);
}

std::vector<float> HeightScanUpdater::get() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return exclude_under_body(height_scan_);
}

std::vector<float> HeightScanUpdater::exclude_under_body(const std::vector<float>& scan)
{
    // Grid is centered at the LiDAR; convert cell xy → base frame and keep
    // only cells outside the same rectangle as height_scan_excluding_body.
    if (static_cast<int>(scan.size()) != kHeightScanRawSize)
    {
        return make_default_height_scan();
    }

    std::vector<float> out;
    out.reserve(kHeightScanSize);
    const float half_x = 0.5f * kHeightScanSizeX;
    const float half_y = 0.5f * kHeightScanSizeY;
    const float eps = kHeightScanResolution * 1.0e-4f;

    for (int ix = 0; ix < kHeightScanGridNx; ++ix)
    {
        for (int iy = 0; iy < kHeightScanGridNy; ++iy)
        {
            const float x_cell = -half_x + static_cast<float>(ix) * kHeightScanResolution;
            const float y_cell = -half_y + static_cast<float>(iy) * kHeightScanResolution;
            const float x_base = kGridCenterOffsetX + x_cell;
            const float y_base = kGridCenterOffsetY + y_cell;
            const bool under_body =
                std::abs(x_base) <= kExcludeHalfExtentX + eps
                && std::abs(y_base) <= kExcludeHalfExtentY + eps;
            if (!under_body)
            {
                out.push_back(scan[ix * kHeightScanGridNy + iy]);
            }
        }
    }
    if (static_cast<int>(out.size()) != kHeightScanSize)
    {
        return make_default_height_scan();
    }
    return out;
}

void HeightScanUpdater::on_height_scan(const sensor_msgs::msg::dds_::PointCloud2_& msg)
{
    std::vector<float> scan;
    if (!parse_height_scan(msg, scan))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    height_scan_ = std::move(scan);
}

bool HeightScanUpdater::parse_height_scan(
    const sensor_msgs::msg::dds_::PointCloud2_& msg,
    std::vector<float>& out)
{
    if (msg.height() != static_cast<uint32_t>(kHeightScanGridNx)
        || msg.width() != static_cast<uint32_t>(kHeightScanGridNy))
    {
        return false;
    }

    const uint32_t point_step = msg.point_step();
    if (point_step < sizeof(float)
        || msg.data().size() < static_cast<size_t>(kHeightScanRawSize) * point_step)
    {
        return false;
    }

    int z_offset = 0;
    for (const auto& field : msg.fields())
    {
        if (field.name() == "z")
        {
            z_offset = static_cast<int>(field.offset());
            break;
        }
    }

    out.resize(kHeightScanRawSize);
    const uint8_t* buffer = msg.data().data();
    for (int i = 0; i < kHeightScanRawSize; ++i)
    {
        std::memcpy(
            &out[i],
            buffer + static_cast<size_t>(i) * point_step + static_cast<size_t>(z_offset),
            sizeof(float));
    }
    return true;
}

} // namespace go2
