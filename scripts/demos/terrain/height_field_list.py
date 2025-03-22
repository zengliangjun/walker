from isaaclab.terrains.height_field import hf_terrains_cfg

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

HEIGHTFIELD_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "PyramidStairs": hf_terrains_cfg.HfPyramidStairsTerrainCfg(
            proportion=0.2,
            border_width=0.5,

            step_height_range=(0.1, 0.5),
            step_width=0.5,
        ),
        "InvertedPyramidStairs": hf_terrains_cfg.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            border_width=0.5,

            step_height_range=(0.1, 0.5),
            step_width=0.5,
        ),
        "DiscreteObstacles": hf_terrains_cfg.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            border_width=0.5,

            obstacle_width_range=(0.5, 3.5),
            obstacle_height_range=(0.5, 3.5),
            num_obstacles=3,
        ),
        "WaveTerrain": hf_terrains_cfg.HfWaveTerrainCfg(
            proportion=0.2,
            border_width=0.5,

            amplitude_range=(0.5, 3.5),
            num_waves=1,
        ),
        "SteppingStones": hf_terrains_cfg.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            border_width=0.5,
            stone_height_max=3.5,
            stone_width_range=(0.5, 3.5),
            stone_distance_range=(0.5, 3.5),
            holes_depth=-10,
        ),

    },
)
"""Rough terrains configuration."""