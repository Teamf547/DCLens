"""Compatibility imports for the original simulation module.

Implementations live in the separately named pipeline stages. Both import
paths expose the same function objects and retain the original signatures.
"""
from .buffer_size_probe import BufferSizeProbe
from .mag_map_sim import MagMapSim

__all__ = ['BufferSizeProbe', 'MagMapSim']
