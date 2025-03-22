from __future__ import annotations
from typing import TYPE_CHECKING

from isaaclab import terrains
import numpy as np
import os.path as osp

from isaaclab import terrains
import trimesh
from isaaclab.utils.dict import dict_to_md5_hash
from isaaclab.utils.io import dump_yaml
import pickle


if TYPE_CHECKING:
    from isaaclabex.terrains import ext_generator_cfg

class TerrainGenerator(terrains.TerrainGenerator):

    terrain_goals: dict[str, np.ndarray]
    """The goals of each sub-terrain. str is (num_rows_num_cols)."""
    def __init__(self, cfg: ext_generator_cfg.TerrainGeneratorCfg, device: str = "cpu"):

        self.terrain_goals = {}
        super(TerrainGenerator, self).__init__(cfg, device)

        _half_rows = -self.cfg.size[0] * self.cfg.num_rows * 0.5
        _half_cols = -self.cfg.size[1] * self.cfg.num_cols * 0.5
        for _key, _goals in self.terrain_goals.items():
            _goals[:, 0] += _half_rows
            _goals[:, 1] += _half_cols


    def _generate_random_terrains(self):
        """Add terrains based on randomly sampled difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # randomly sample sub-terrains
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # coordinate index of the sub-terrain
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            # randomly sample terrain index
            sub_index = self.np_rng.choice(len(proportions), p=proportions)
            # randomly sample difficulty parameter
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            # generate terrain
            _cfg = sub_terrains_cfgs[sub_index].copy()
            _cfg.row = sub_row
            _cfg.col = sub_col

            mesh, origin = self._get_terrain_mesh(difficulty, _cfg)
            # add to sub-terrains
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    def _generate_curriculum_terrains(self):
        """Add terrains based on the difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        # find the sub-terrain index for each column
        # we generate the terrains based on their proportion (not randomly sampled)
        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # curriculum-based sub-terrains
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                # vary the difficulty parameter linearly over the number of rows
                # note: based on the proportion, multiple columns can have the same sub-terrain type.
                #  Thus to increase the diversity along the rows, we add a small random value to the difficulty.
                #  This ensures that the terrains are not exactly the same. For example, if the
                #  the row index is 2 and the number of rows is 10, the nominal difficulty is 0.2.
                #  We add a small random value to the difficulty to make it between 0.2 and 0.3.
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                # generate terrain
                _cfg = sub_terrains_cfgs[sub_indices[sub_col]].copy()
                _cfg.row = sub_row
                _cfg.col = sub_col
                mesh, origin = self._get_terrain_mesh(difficulty, _cfg)
                # add to sub-terrains
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_indices[sub_col]])


    def _get_terrain_mesh(self, difficulty: float, _cfgOrg) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Generate a sub-terrain mesh based on the input difficulty parameter.

        If caching is enabled, the sub-terrain is cached and loaded from the cache if it exists.
        The cache is stored in the cache directory specified in the configuration.

        .. Note:
            This function centers the 2D center of the mesh and its specified origin such that the
            2D center becomes :math:`(0, 0)` instead of :math:`(size[0] / 2, size[1] / 2).

        Args:
            difficulty: The difficulty parameter.
            cfg: The configuration of the sub-terrain.

        Returns:
            The sub-terrain mesh and origin.
        """
        # copy the configuration
        cfg = _cfgOrg.copy()
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # generate hash for the sub-terrain
        sub_terrain_hash = dict_to_md5_hash(cfg.to_dict())
        # generate the file name
        sub_terrain_cache_dir = osp.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_obj_filename = osp.join(sub_terrain_cache_dir, "mesh.obj")
        sub_terrain_csv_filename = osp.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = osp.join(sub_terrain_cache_dir, "cfg.yaml")
        sub_terrain_goal_filename = osp.join(sub_terrain_cache_dir, "goal.pkl")

        # check if hash exists - if true, load the mesh and origin and return
        if self.cfg.use_cache and osp.exists(sub_terrain_obj_filename):
            # load existing mesh
            mesh = trimesh.load_mesh(sub_terrain_obj_filename, process=False)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")
            if osp.exists(sub_terrain_goal_filename):
                with open(sub_terrain_goal_filename, "rb") as _f:
                    goals = pickle.load(_f)
                    _key = f'{_cfgOrg.row}_{_cfgOrg.col}'
                    self.terrain_goals[_key] = goals
            # return the generated mesh
            return mesh, origin

        # generate the terrain
        _result = cfg.function(difficulty, cfg)
        if isinstance(_result, tuple):
            if len(_result) == 2:
                meshes, origin = _result
            else:
                meshes, origin, goals = _result
        elif isinstance(_result, dict):
            meshes = _result["meshes"]
            origin = _result["origin"]
            if "goals" in _result:
                goals = _result["goals"]                
            else:
                goals = None

        mesh = trimesh.util.concatenate(meshes)
        # offset mesh such that they are in their center
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        # change origin to be in the center of the sub-terrain
        origin += transform[0:3, -1]

        if goals is not None:
            #goals += transform[0:3, -1]
            goals[:, 0] += (_cfgOrg.row + 0.5 - 0.5) * self.cfg.size[0]
            goals[:, 1] += (_cfgOrg.col + 0.5 - 0.5) * self.cfg.size[1]

            _key = f'{_cfgOrg.row}_{_cfgOrg.col}'
            self.terrain_goals[_key] = goals

        # if caching is enabled, save the mesh and origin
        if self.cfg.use_cache:
            # create the cache directory
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)
            # save the data
            mesh.export(sub_terrain_obj_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, cfg)

            if goals is not None:
                with open(sub_terrain_goal_filename, "wb") as _f:
                    pickle.dump(goals, _f)

        # return the generated mesh
        return mesh, origin
