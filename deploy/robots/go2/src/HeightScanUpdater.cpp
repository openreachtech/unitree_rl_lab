#include "HeightScanUpdater.h"
#include "param.h"

#include <algorithm>
#include <cmath>
#include <numeric>
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
    // config.yaml's FSM.Velocity.height_scan block already carries flat_override /
    // flat_value (see State_RLBase.cpp's REGISTER_OBSERVATION(height_scan)); the grid
    // shape lives alongside it so both are configured in one place per deployed policy.
    const auto hs_cfg = param::config["FSM"]["Velocity"]["height_scan"];
    if (hs_cfg.IsDefined())
    {
        if (hs_cfg["grid_nx"].IsDefined()) cfg_.grid_nx = hs_cfg["grid_nx"].as<int>();
        if (hs_cfg["grid_ny"].IsDefined()) cfg_.grid_ny = hs_cfg["grid_ny"].as<int>();
        if (hs_cfg["resolution"].IsDefined()) cfg_.resolution = hs_cfg["resolution"].as<float>();
        if (hs_cfg["x_min"].IsDefined()) cfg_.x_min = hs_cfg["x_min"].as<float>();
        if (hs_cfg["y_min"].IsDefined()) cfg_.y_min = hs_cfg["y_min"].as<float>();
        if (hs_cfg["exclude_half_x"].IsDefined()) cfg_.exclude_half_x = hs_cfg["exclude_half_x"].as<float>();
        if (hs_cfg["exclude_half_y"].IsDefined()) cfg_.exclude_half_y = hs_cfg["exclude_half_y"].as<float>();
        if (hs_cfg["gather_enabled"].IsDefined()) cfg_.gather_enabled = hs_cfg["gather_enabled"].as<bool>();
    }

    const int raster_size = cfg_.grid_nx * cfg_.grid_ny;
    keep_index_.clear();
    if (!cfg_.gather_enabled)
    {
        keep_index_.resize(raster_size);
        std::iota(keep_index_.begin(), keep_index_.end(), 0);
    }
    else
    {
        keep_index_.reserve(raster_size);
        const float eps = cfg_.resolution * 1.0e-4f;
        for (int ix = 0; ix < cfg_.grid_nx; ++ix)
        {
            const float x = cfg_.x_min + static_cast<float>(ix) * cfg_.resolution;
            for (int iy = 0; iy < cfg_.grid_ny; ++iy)
            {
                const float y = cfg_.y_min + static_cast<float>(iy) * cfg_.resolution;
                const bool under_body =
                    std::fabs(x) <= cfg_.exclude_half_x + eps &&
                    std::fabs(y) <= cfg_.exclude_half_y + eps;
                if (!under_body)
                {
                    keep_index_.push_back(ix * cfg_.grid_ny + iy);
                }
            }
        }
    }
    height_scan_.assign(keep_index_.size(), kHeightScanEmpty);

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
    spdlog::info(
        "HeightScanUpdater: {}x{} raster ({} cells), gather_enabled={}, policy input size={}",
        cfg_.grid_nx, cfg_.grid_ny, raster_size, cfg_.gather_enabled, keep_index_.size());
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

    // executor_.spin() runs this callback on a single dedicated thread (see init()),
    // so this flag needs no synchronization of its own.
    static bool first_received = false;
    if (!first_received)
    {
        first_received = true;
        const auto non_empty = std::count_if(scan.begin(), scan.end(),
            [](float v) { return v != kHeightScanEmpty; });
        spdlog::info("HeightScanUpdater: first HeightMap received on {} ({}/{} non-empty cells)",
            kHeightScanTopic, non_empty, scan.size());
    }

    std::lock_guard<std::mutex> lock(mutex_);
    height_scan_ = std::move(scan);
}

bool HeightScanUpdater::parse_height_scan(
    const heightmap_generator::msg::HeightMap& msg,
    std::vector<float>& out) const
{
    // heightmap_generator publishes the full raster (msg.width/height describe it);
    // the policy only ever trained on the cells keep_index_ selects out of it (or the
    // whole raster, if gather_enabled is false), so validate against the raster
    // shape, not keep_index_.size().
    const int raster_size = cfg_.grid_nx * cfg_.grid_ny;
    if (msg.width != static_cast<uint32_t>(cfg_.grid_nx)
        || msg.height != static_cast<uint32_t>(cfg_.grid_ny)
        || std::abs(msg.resolution - cfg_.resolution) > 1.0e-5f
        || std::abs(msg.x_min - cfg_.x_min) > 1.0e-5f
        || std::abs(msg.y_min - cfg_.y_min) > 1.0e-5f
        || msg.data.size() != static_cast<size_t>(raster_size))
    {
        return false;
    }

    out.resize(keep_index_.size());
    for (std::size_t i = 0; i < keep_index_.size(); ++i)
    {
        const float value = msg.data[static_cast<std::size_t>(keep_index_[i])];
        // The contract defines unknown cells as 0.0. The generator can leave cells
        // without observed neighbours as NaN, which must not reach ONNX.
        out[i] = std::isfinite(value) ? value : kHeightScanEmpty;
    }
    return true;
}

} // namespace go2
