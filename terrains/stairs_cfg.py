import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

G1_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),        # 8.0 だと段数が5段程度。10.0で7〜8段取れる
    border_width=20.0,
    num_rows=10,              # = difficulty レベル数（カリキュラムの段数）
    num_cols=20,              # = 各レベルのバリエーション数
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,          # row方向に難易度が上がる
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        # 平地を残すと「平地フォーム」が壊れにくい
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.03, 0.17),   # difficulty=0→0.03m, 1→0.17m で線形補間
            step_width=0.30,                  # 踏面
            platform_width=3.0,               # 中央の平坦部（スポーン地点）
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.03, 0.17),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)