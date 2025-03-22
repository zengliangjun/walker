import isaaclab.terrains.trimesh.mesh_terrains_cfg as mesh_terrains_cfg

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

MESH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "box_rails": mesh_terrains_cfg.MeshRailsTerrainCfg(
            proportion=0.1,
            rail_thickness_range=(0.2, 0.6),
            rail_height_range=(0.2, 0.6),
            platform_width=2,
        ),
        "pit": mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=0.1,
            pit_depth_range=(0.05, 1.5),
            double_pit=False,
            platform_width=2,
        ),
        "double_pit": mesh_terrains_cfg.MeshPitTerrainCfg(
            proportion=0.1,
            pit_depth_range=(0.05, 1.5),
            double_pit=True,
            platform_width=2,
        ),
        "boxes": mesh_terrains_cfg.MeshBoxTerrainCfg(
            proportion=0.1,
            box_height_range=(0.05, 1.5),
            double_box=True,
            platform_width=2,
        ),
        "gap": mesh_terrains_cfg.MeshGapTerrainCfg(
            proportion=0.1,
            gap_width_range=(0.3, 1.5),
            platform_width=2,
        ),
        "ring": mesh_terrains_cfg.MeshFloatingRingTerrainCfg(
            proportion=0.1,
            ring_width_range=(0.3, 0.5),
            ring_height_range=(0.3, 0.5),
            ring_thickness=0.5,
            platform_width=2,
        ),
        "star": mesh_terrains_cfg.MeshStarTerrainCfg(
            proportion=0.1,
            num_bars=5,
            bar_width_range=(0.1, 0.5),
            bar_height_range=(0.1, 0.5),
            platform_width=4.0,
        ),
        "RepeatedCylinders": mesh_terrains_cfg.MeshRepeatedCylindersTerrainCfg(
            proportion=0.1,
            object_params_start=mesh_terrains_cfg.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=1,
                height=3.2,
                radius=0.5,
                max_yx_angle=15.0,
                degrees=True,
            ),
            object_params_end=mesh_terrains_cfg.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=3,
                height=6.2,
                radius=1.5,
                max_yx_angle=45,
                degrees=True,
            ),
            max_height_noise=0.0,
            platform_width=0.0
        ),
        "RepeatedPyramids": mesh_terrains_cfg.MeshRepeatedPyramidsTerrainCfg(
            proportion=0.1,
            object_params_start=mesh_terrains_cfg.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=1,
                height=3.2,
                radius=0.5,
                max_yx_angle=15.0,
                degrees=True,
            ),
            object_params_end=mesh_terrains_cfg.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=3,
                height=6.2,
                radius=1.5,
                max_yx_angle=45,
                degrees=True,
            ),
            max_height_noise=0.0,
            platform_width=0.0
        ),
        "RepeatedBoxes": mesh_terrains_cfg.MeshRepeatedBoxesTerrainCfg(
            proportion=0.1,
            object_params_start=mesh_terrains_cfg.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=1,
                height=3.2,
                size=(1.5, 1.5),
                max_yx_angle=5.0,
                degrees=True,
            ),
            object_params_end=mesh_terrains_cfg.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=3,
                height=6.2,
                size=(3.5, 3.5),
                max_yx_angle=15,
                degrees=True,
            ),
            max_height_noise=0.2,
            platform_width=0.0
        ),
    },
)
"""Rough terrains configuration."""