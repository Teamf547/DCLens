"""MagMapSim: coordinate GPU stages and write one map per lens snapshot."""
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from .config import _resolve_config
from .io import _npy_path, _write_json
from .lens import Lens
from .logger import configure
from .magnification_map import MagnificationMap, MagnificationMapSeries, _COORDINATES
from .physics.pm_builder import _p3m_plane_metadata
from ._pipeline import _bounds, _prepare_planes, _upload_planes

__all__ = ['MagMapSim']


def _simulate_frame(engine, cfg, geometry, bounds):
    img_xmin, img_xmax, img_ymin, img_ymax = bounds
    delta = cfg.max_coarse_per_chunk * cfg.coarse_ps
    nx_chunks = int(np.ceil((img_xmax-img_xmin)/delta))
    ny_chunks = int(np.ceil((img_ymax-img_ymin)/delta))
    result = np.zeros((geometry['ny'], geometry['nx']), dtype=np.float32)
    total = nx_chunks*ny_chunks
    for iy in range(ny_chunks):
        for ix in range(nx_chunks):
            chunk = engine.simulate_chunk(
                img_xmin+ix*delta, min(img_xmin+(ix+1)*delta, img_xmax),
                img_ymin+iy*delta, min(img_ymin+(iy+1)*delta, img_ymax),
                cfg.coarse_ps, cfg.finest_ps, cfg.distortion_threshold,
                cfg.max_refine_iters,
                geometry['src_xmin'], geometry['src_xmax'],
                geometry['src_ymin'], geometry['src_ymax'],
                geometry['nx'], geometry['ny'], cfg.ss)
            # Preserve the original float32, image-chunk accumulation order.
            result += chunk
            del chunk
            completed = iy*nx_chunks + ix + 1
            if cfg.show_progress and (completed == total or completed % max(1,total//10) == 0):
                print(f'[IPM] {completed}/{total} chunks', flush=True)
    return result


def MagMapSim(lens, *, config=None, output=None, save_dir=None,
              keep_in_memory=None, **keywords):
    """Simulate a single map or stream a lens sequence through one CUDA engine.

    Angles/pixel scales are arcsec. Sequence results always contain paths only;
    single results retain their existing float32 array by default.
    Set deflector='pp' for GPU whole-plane summation without PM/FFT/buckets;
    deflector='p3m' retains the mesh plus local correction method.
    """
    if not isinstance(lens, Lens):
        raise TypeError('MagMapSim requires a Lens')
    cfg = _resolve_config(config, keywords)
    configure(cfg.verbose)
    sequence = lens.mode == 'sequence'
    if sequence and output is not None:
        raise ValueError('Sequence output uses save_dir and deterministic frame names')
    if sequence and keep_in_memory:
        raise ValueError('Sequence results store paths only; keep_in_memory must be False')
    keep = not sequence if keep_in_memory is None else bool(keep_in_memory)
    directory = Path(save_dir or ('maps' if sequence else '.')).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory/'manifest.json'
    if sequence and (manifest_path.exists() or any(directory.glob('map_*.npy'))):
        raise FileExistsError('Choose a new save_dir for this sequence')
    cx, cy = cfg.img_center
    width, height = cfg.src_considered_region
    nx, ny = int(np.ceil(width/cfg.srcp_ps)), int(np.ceil(height/cfg.srcp_ps))
    xmin, ymin = cx-width/2, cy-height/2
    geometry = dict(src_xmin=xmin, src_xmax=xmin+nx*cfg.srcp_ps,
                    src_ymin=ymin, src_ymax=ymin+ny*cfg.srcp_ps,
                    srcp_ps=cfg.srcp_ps, nx=nx, ny=ny,
                    coordinate_convention=_COORDINATES)
    parameters = dict(asdict(cfg), D_L_pc=lens.D_L_pc,
                      D_S_pc=lens.D_S_pc, N_planes=lens.N_planes)
    bounds = _bounds(cfg, cfg.buffer_size)
    frames = []
    from . import _cuda
    engine = _cuda.Engine(cfg.max_active_cells, cfg.verbose)
    try:
        for i, lens_file in enumerate(lens.files):
            start = time.perf_counter()
            table = lens.load_frame(i)
            input_sec = time.perf_counter()-start
            setup_start = time.perf_counter()
            planes = _prepare_planes(table, lens, cfg, bounds)
            plane_metadata = [dict(z_pc=float(p['z_pc']),
                                   lens_count=len(p['lenses']['x_pc'])) for p in planes]
            p3m_metadata = (dict(enabled=True, assignment='TSC', assignment_order_p=2,
                                 finite_difference_beta=4/3, particle_shape='S1',
                                 a_pm_over_H=cfg.p3m_a_pm_over_H,
                                 padding_factor=cfg.p3m_padding_factor,
                                 planes=_p3m_plane_metadata(planes))
                             if cfg.deflector == 'p3m' else dict(enabled=False))
            _upload_planes(engine, planes, lens, cfg)
            del planes, table
            setup_sec = time.perf_counter()-setup_start
            simulate_start = time.perf_counter()
            array = _simulate_frame(engine, cfg, geometry, bounds)
            simulate_sec = time.perf_counter()-simulate_start
            path = directory/f'map_{i:06d}.npy' if sequence else _npy_path(directory/(output or 'map.npy'))
            path.parent.mkdir(parents=True, exist_ok=True)
            save_start = time.perf_counter()
            np.save(path, array, allow_pickle=False)
            metadata = dict(
                geometry=geometry,
                lens=dict(input_file=str(lens_file), frame_index=i),
                simulation=parameters,
                deflector=dict(method=cfg.deflector, planes=plane_metadata,
                               min_r2_pc2=1e-12),
                p3m=p3m_metadata,
                timing=dict(input_sec=input_sec, deflector_setup_sec=setup_sec,
                            p3m_sec=setup_sec if cfg.deflector == 'p3m' else 0.0,
                            ipm_raster_sec=simulate_sec,
                            save_sec=time.perf_counter()-save_start,
                            total_sec=time.perf_counter()-start))
            metadata_path = path.with_suffix('.json')
            _write_json(metadata_path, metadata)
            if sequence:
                frames.append(dict(index=i, lens_file=str(lens_file),
                                   map_file=path.name, metadata_file=metadata_path.name))
                del array
            else:
                result = MagnificationMap(array=array if keep else None, path=path,
                                          metadata=metadata, metadata_path=metadata_path)
                del array
            if cfg.show_progress:
                print(f'[Output] Frame {i+1}/{lens.frame_count}: {path}', flush=True)
    finally:
        engine.close()
    if sequence:
        _write_json(manifest_path, dict(format_version=1, frames=frames,
                                       geometry=geometry, simulation=parameters))
        return MagnificationMapSeries(manifest_path)
    return result
