import torch

def visualize_heightmap(env, heightmap, lidar_processor):
    """ハイトマップを描画する"""
    if not hasattr(env, "_heightmap_marker"):
        from isaaclab.markers.visualization_markers import VisualizationMarkers
        from isaaclab.markers.config import CUBOID_MARKER_CFG
        
        cfg = CUBOID_MARKER_CFG.copy()
        cfg.prim_path = "/Visuals/HeightmapPoints"
        cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
        cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
        env._heightmap_marker = VisualizationMarkers(cfg)
        
        nx, ny = lidar_processor.nx, lidar_processor.ny
        res = lidar_processor.res
        ix = torch.arange(nx, device=env.device)
        iy = torch.arange(ny, device=env.device)
        grid_x, grid_y = torch.meshgrid(ix, iy, indexing='ij')
        env._grid_x_local = lidar_processor.x_range[0] + grid_x.flatten() * res
        env._grid_y_local = lidar_processor.y_range[0] + grid_y.flatten() * res
        
    h = heightmap[0]

    if hasattr(lidar_processor, "last_empty_mask") and lidar_processor.last_empty_mask is not None:
        valid_idx = ~lidar_processor.last_empty_mask[0]
    else:
        valid_idx = h != lidar_processor.unknown_fill
        
    if valid_idx.any():
        h_valid = h[valid_idx]
        x_loc = env._grid_x_local[valid_idx]
        y_loc = env._grid_y_local[valid_idx]
        
        qw, qx, qy, qz = env.scene["robot"].data.root_quat_w[0]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        x_w = env.scene["robot"].data.root_pos_w[0, 0] + x_loc * cos_yaw - y_loc * sin_yaw
        y_w = env.scene["robot"].data.root_pos_w[0, 1] + x_loc * sin_yaw + y_loc * cos_yaw
        z_w = env.scene["robot"].data.root_pos_w[0, 2] - h_valid - lidar_processor.offset
        
        points = torch.stack([x_w, y_w, z_w], dim=-1)
        env._heightmap_marker.visualize(translations=points)
