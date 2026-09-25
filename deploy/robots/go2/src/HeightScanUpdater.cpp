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
    // Silence is indistinguishable from flat ground once the map is in the observation:
    // height_scan_ starts at the flat-ground fill, so a policy that never receives a
    // message walks into walls exactly as if the terrain were empty. Say so rather than
    // letting that look like a locomotion failure.
    if (msg_count_ == 0)
    {
        static bool warned = false;
        if (!warned)
        {
            spdlog::warn(
                "HeightScanUpdater: no message on {} yet -- the policy is being fed flat "
                "ground ({:.4f} everywhere). Is the simulator/mapper publishing?",
                kHeightScanTopic, kHeightScanFlatDefault);
            warned = true;
        }
    }
    return height_scan_;
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
    ++msg_count_;
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
