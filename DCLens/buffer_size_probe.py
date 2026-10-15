"""BufferSizeProbe: estimate the image-plane margin over lens snapshots."""
import numpy as np

from .config import _resolve_config
from .lens import Lens
from .logger import configure, logger
from ._pipeline import _bounds, _prepare_planes, _upload_planes

__all__ = ['BufferSizeProbe']


def BufferSizeProbe(lens, *, config=None, buffer_step=.01,
                    max_buffer_steps=200, return_frames=False, **keywords):
    """Probe all snapshots; return the maximum recommended buffer in arcsec.

    Select GPU deflection with deflector='p3m' (default) or deflector='pp'.
    The eight-point boundary probe is the established heuristic, not a proof
    that every lensed image lies inside the resulting image-plane region.
    """
    if not isinstance(lens, Lens):
        raise TypeError('BufferSizeProbe requires a Lens')
    if not np.isfinite(buffer_step) or buffer_step <= 0 or not isinstance(
            max_buffer_steps, int) or max_buffer_steps < 1:
        raise ValueError('Require positive buffer_step and integer max_buffer_steps')
    from . import _cuda
    cfg = _resolve_config(config, keywords)
    configure(cfg.verbose)
    xc, yc = cfg.img_center
    width, height = cfg.src_considered_region
    bounds = _bounds(cfg, max_buffer_steps*buffer_step)
    recommended = []
    # No rasterizer or IPM queues are allocated by the probe.
    engine = _cuda.Engine(cfg.max_active_cells, cfg.verbose)
    try:
        for i in range(lens.frame_count):
            table = lens.load_frame(i)
            planes = _prepare_planes(table, lens, cfg, bounds)
            _upload_planes(engine, planes, lens, cfg)
            del planes, table
            value = engine.estimate_buffer_size(
                xc, yc, width, height, *_bounds(cfg, 0),
                buffer_step, max_buffer_steps)
            recommended.append(value)
            if value >= buffer_step*max_buffer_steps:
                logger.warning('[Probe] Frame %d reached probe limit; increase max_buffer_steps', i)
    finally:
        engine.close()
    value = max(recommended)
    return {'recommended': value, 'frames': recommended} if return_frames else value


