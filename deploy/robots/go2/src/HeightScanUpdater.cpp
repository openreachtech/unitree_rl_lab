#include "HeightScanUpdater.h"

#include <cmath>
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
    node_ = rclcpp::Node::make_shared("go2_heightmap_receiver");
    height_scan_sub_ = node_->create_subscription<heightmap_generator::msg::HeightMap>(
        kHeightScanTopic,
        rclcpp::QoS(10),
        [this](heightmap_generator::msg::HeightMap::SharedPtr msg) {
            on_height_scan(std::move(msg));
        });
    executor_.add_node(node_);
    spin_thread_ = std::thread([this]() { executor_.spin(); });
    spin_thread_.detach();
    spdlog::info("HeightScanUpdater subscribed to ROS 2 {} (heightmap_generator/HeightMap)", kHeightScanTopic);
}

std::vector<float> HeightScanUpdater::get() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return height_scan_;
}

void HeightScanUpdater::on_height_scan(const heightmap_generator::msg::HeightMap::SharedPtr msg)
{
    std::vector<float> scan;
    if (!parse_height_scan(*msg, scan))
    {
        spdlog::warn("Rejected incompatible HeightMap on {}", kHeightScanTopic);
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    height_scan_ = std::move(scan);
}

bool HeightScanUpdater::parse_height_scan(
    const heightmap_generator::msg::HeightMap& msg,
    std::vector<float>& out)
{
    if (msg.width != static_cast<uint32_t>(kHeightScanGridNx)
        || msg.height != static_cast<uint32_t>(kHeightScanGridNy)
        || std::abs(msg.resolution - kHeightScanResolution) > 1.0e-5f
        || std::abs(msg.x_min - kHeightScanXMin) > 1.0e-5f
        || std::abs(msg.y_min - kHeightScanYMin) > 1.0e-5f
        || msg.data.size() != static_cast<size_t>(kHeightScanSize))
    {
        return false;
    }
    out = msg.data;
    // The contract defines unknown cells as 0.0. The current generator can leave
    // cells without observed neighbours as NaN, which must not reach ONNX.
    for (float& value : out)
    {
        if (!std::isfinite(value))
        {
            value = kHeightScanEmpty;
        }
    }
    return true;
}

} // namespace go2
