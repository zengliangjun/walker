from __future__ import annotations
from typing import TYPE_CHECKING
from isaaclab import terrains
import isaaclab.sim as sim_utils
from isaaclab.markers import config
from isaaclab.markers import VisualizationMarkers
import torch
import numpy as np

if TYPE_CHECKING:
    from isaaclabex.terrains import ext_generator, ext_importer_cfg


class TerrainImporter(terrains.TerrainImporter):

    def __init__(self, cfg: ext_importer_cfg.TerrainImporterCfg):
        if cfg.terrain_type != "generator":
            super(TerrainImporter, self).__init__(cfg)
            return

        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create a dict of meshes
        self.meshes = dict()
        self.warp_meshes = dict()
        self.env_origins = None
        self.terrain_origins = None
        # private variables
        self._terrain_flat_patches = dict()

        # auto-import the terrain based on the config

        # check config is provided
        if self.cfg.terrain_generator is None:
            raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
        # generate the terrain
        terrain_generator = ext_generator.TerrainGenerator(cfg=self.cfg.terrain_generator, device=self.device)
        self.import_mesh("terrain", terrain_generator.terrain_mesh)
        # configure the terrain origins based on the terrain generator
        self.configure_env_origins(terrain_generator.terrain_origins)
        # refer to the flat patches
        self._terrain_flat_patches = terrain_generator.flat_patches

        self.terrain_goals = terrain_generator.terrain_goals


        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def set_debug_vis(self, debug_vis: bool) -> bool:
        super(TerrainImporter, self).set_debug_vis(debug_vis)

        if debug_vis:
            if not hasattr(self, "goals_visualizer"):
                _cfg = config.POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/TerrainGoals")
                for _key in _cfg.markers:
                    _cfg.markers[_key].radius = 0.15
                    _cfg.markers[_key].visible = True
                    _cfg.markers[_key].visual_material.opacity = 0.6

                self.goals_visualizer = VisualizationMarkers(cfg=_cfg)

                _translations = []
                _marker_indices = []
                for _key in self.terrain_goals:
                    _goals = self.terrain_goals[_key]
                    _index = torch.ones(_goals.shape[0], dtype=torch.int32)
                    _index[0] = 0
                    _index[-1] = 2

                    _translations.append(_goals)
                    _marker_indices.append(_index)

                _translations = np.concatenate(_translations, axis=0)
                _marker_indices = torch.cat(_marker_indices, dim=0)

                # set the marker
                self.goals_visualizer.visualize(translations=_translations, marker_indices=_marker_indices)

            self.goals_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goals_visualizer"):
                self.goals_visualizer.set_visibility(False)
        # report success
        return True
