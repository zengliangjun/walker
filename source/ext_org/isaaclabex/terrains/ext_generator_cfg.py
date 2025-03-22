from isaaclab.utils import configclass
from dataclasses import MISSING
from collections.abc import Callable

from isaaclab import terrains



@configclass
class SubTerrainBaseCfg(terrains.SubTerrainBaseCfg):
    """Base class for terrain configurations.

    All the sub-terrain configurations must inherit from this class.

    The :attr:`size` attribute is the size of the generated sub-terrain. Based on this, the terrain must
    extend from :math:`(0, 0)` to :math:`(size[0], size[1])`.
    """

    function: Callable[[float, terrains.SubTerrainBaseCfg], dict] = MISSING

@configclass
class TerrainGeneratorCfg(terrains.TerrainGeneratorCfg):

    sub_terrains: dict[str, SubTerrainBaseCfg] = MISSING
    """Dictionary of sub-terrain configurations.

    The keys correspond to the name of the sub-terrain configuration and the values are the corresponding
    configurations.
    """
