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
    // Grid is centered at the LiDAR; convert cell xy → base frame, then mask
    // the same rectangle as height_scan_excluding_body.
    std::vector<float> out = scan;
    if (static_cast<int>(out.size()) != kHeightScanSize)
    {
        return out;
    }

    const float half_x = 0.5f * kHeightScanSizeX;
    const float half_y = 0.5f * kHeightScanSizeY;

    for (int ix = 0; ix < kHeightScanGridNx; ++ix)
    {
        for (int iy = 0; iy < kHeightScanGridNy; ++iy)
        {
            const float x_cell = -half_x + static_cast<float>(ix) * kHeightScanResolution;
            const float y_cell = -half_y + static_cast<float>(iy) * kHeightScanResolution;
            const float x_base = kLidarOffsetX + x_cell;
            const float y_base = kLidarOffsetY + y_cell;
            if (std::abs(x_base) <= kExcludeHalfExtentX
                && std::abs(y_base) <= kExcludeHalfExtentY)
            {
                out[ix * kHeightScanGridNy + iy] = kExcludeFillValue;
            }
        }
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
        || msg.data().size() < static_cast<size_t>(kHeightScanSize) * point_step)
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

    out.resize(kHeightScanSize);
    const uint8_t* buffer = msg.data().data();
    for (int i = 0; i < kHeightScanSize; ++i)
    {
        std::memcpy(
            &out[i],
            buffer + static_cast<size_t>(i) * point_step + static_cast<size_t>(z_offset),
            sizeof(float));
    }
    return true;
}

} // namespace go2
