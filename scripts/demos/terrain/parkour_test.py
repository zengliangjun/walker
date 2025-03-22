from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclabex.terrains.trimesh import parkour_mesh_cfg

'''
        "parkour_hf": parkour_hf_cfg.ParkourHurdleTerrainCfg(
            proportion=0.2,
            noise_range=(0.02, 0.02),
            noise_step=0.02,
            num_stones = 6,
            hurdle_y = 2.5,
            hurdle_x_range = [0.2, 0.4],
            hurdle_z_range = (0.15, 0.3),
            hurdle_z_noise = (-0.1, 0.1),
            hurdle_distance_x_range = [1.2, 2.2],
            hurdle_distance_y_range = [-0.4, 0.4],
            platform_space = 2.5,
            platform_height = 0,
        ),
'''

PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(18.0, 4.0),  # X Y
    border_width=0.0,
    num_rows=2, #4,
    num_cols=2, #6,
    horizontal_scale=0.04,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,


    sub_terrains={
        "parkour_mesh": parkour_mesh_cfg.MeshRepeatedHurdleTerrainCfg(
            proportion=0.2,
            platform_x_space=2.5,
            object_params_start=parkour_mesh_cfg.MeshRepeatedHurdleTerrainCfg.ObjectCfg(
                num_objects=6,
                width=0.2,
                length=2,
                height=0.15,
                distance_x=1.2,
                distance_y=-0.4,
            ),
            object_params_end=parkour_mesh_cfg.MeshRepeatedHurdleTerrainCfg.ObjectCfg(
                num_objects=6,
                width=0.4,
                length=2.5,
                height=0.3,
                distance_x=2.2,
                distance_y=0.4,
            ),

            max_height_noise=0.05,
        ),

    },
)
