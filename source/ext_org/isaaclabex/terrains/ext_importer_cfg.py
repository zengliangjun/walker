from isaaclab import terrains  
from isaaclab.utils import configclass
from isaaclabex.terrains import ext_importer, ext_generator_cfg

@configclass
class TerrainImporterCfg(terrains.TerrainImporterCfg):

    class_type: type = ext_importer.TerrainImporter
    """The class to use for the terrain importer.

    Defaults to :class:`isaaclab.terrains.terrain_importer.TerrainImporter`.
    """

    terrain_generator: ext_generator_cfg.TerrainGeneratorCfg | None = None
    """The terrain generator configuration.

    Only used if ``terrain_type`` is set to "generator".
    """