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
        
    # heightmap_spec.yaml からLiDARの取付位置を動的に読み込む
    extrinsics = cfg['frame']['lidar_extrinsics']
    # OmegaConf が numpy.float64 に対応していないため、明示的に float にキャスト
    trans = tuple(float(x) for x in extrinsics['translation_m'])
    rpy = extrinsics['rpy_deg']

    qw, qx, qy, qz = get_lidar_quat_wxyz(rpy[0], rpy[1], rpy[2])
    
    # heightmap_spec.yaml からLiDARの物理スペックを動的に読み込む
    sensor_cfg = cfg['lidar_sensor']
    channels = int(sensor_cfg['channels'])
    v_fov = tuple(float(x) for x in sensor_cfg['vertical_fov_range_deg'])
    h_fov = tuple(float(x) for x in sensor_cfg['horizontal_fov_range_deg'])
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
