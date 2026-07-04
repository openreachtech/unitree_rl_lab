import math
import yaml
import os
import scipy.spatial.transform as transform
from isaaclab.sensors import RayCasterCfg, patterns

def get_lidar_quat_wxyz(r_deg: float, p_deg: float, y_deg: float) -> tuple:
    rot = transform.Rotation.from_euler('xyz', [r_deg, p_deg, y_deg], degrees=True)
    qx, qy, qz, qw = rot.as_quat()
    return (float(qw), float(qx), float(qy), float(qz))

def get_go2_lidar_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot/base/lidar", config_yaml_path: str = None) -> RayCasterCfg:
    """
    Unitree Go2搭載のL1 LiDARの構成を模倣したIsaac Lab用RayCaster設定。
    ※ prim_path は実際のロボットのベースリンクに合わせて適宜変更してインポートしてください。
    """
    if config_yaml_path is None:
        import unitree_rl_lab.unitree_go2_locomotion_heightmap.lidar_processor as lp
        config_yaml_path = os.path.join(os.path.dirname(os.path.abspath(lp.__file__)), "heightmap_spec.yaml")
        
    with open(config_yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    extrinsics = cfg['frame']['lidar_extrinsics']
    trans = tuple(float(x) for x in extrinsics['translation_m'])
    rpy = extrinsics['rpy_deg']

    r_sim = rpy[0] + 180.0 if rpy[0] < 0 else rpy[0] - 180.0
    p_sim = rpy[1]
    y_sim = rpy[2]
    
    r_sim = float(round(r_sim, 4))
    y_sim = float(round(y_sim, 4))

    qw, qx, qy, qz = get_lidar_quat_wxyz(r_sim, p_sim, y_sim)
    
    sensor_cfg = cfg['lidar_sensor']
    channels = int(sensor_cfg['channels'])
    v_fov = tuple(float(x) for x in sensor_cfg['vertical_fov_range_deg'])
    
    # 実機Go2の物理的マスキング（後方および自己干渉領域の除外）を模擬するため、水平視野角を制限
    # 実機ではロボット自身の胴体に遮られる後方（±120度〜±180度）は完全にマスクされて点群が得られません
    h_fov = (-120.0, 120.0)
    
    h_res = float(sensor_cfg['horizontal_res_deg'])
    max_dist = float(sensor_cfg['max_distance_m'])
    
    return RayCasterCfg(
        prim_path=prim_path,
        offset=RayCasterCfg.OffsetCfg(
            pos=trans,
            rot=(qw, qx, qy, qz),
        ),
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=channels,
            vertical_fov_range=v_fov,
            horizontal_fov_range=h_fov,
            horizontal_res=h_res,
        ),
        max_distance=max_dist,
        debug_vis=True,        # デバッグ用点群描画
    )
