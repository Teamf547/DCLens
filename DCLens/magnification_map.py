"""Lightweight in-memory maps and disk-backed sequence records."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import _read_json

_COORDINATES = 'arcsec; array[row_y,col_x]; origin=lower; bounds=pixel_edges'
_GEOMETRY_KEYS = ('src_xmin', 'src_xmax', 'src_ymin', 'src_ymax',
                  'srcp_ps', 'nx', 'ny', 'coordinate_convention')


@dataclass
class MagnificationMap:
    array: np.ndarray | None = None
    path: Path | str | None = None
    metadata: dict = field(default_factory=dict)
    metadata_path: Path | str | None = None

    def __post_init__(self):
        if self.path is not None:
            self.path = Path(self.path).expanduser().resolve()
        if self.metadata_path is not None:
            self.metadata_path = Path(self.metadata_path).expanduser().resolve()
        elif self.path is not None and self.path.with_suffix('.json').is_file():
            self.metadata_path = self.path.with_suffix('.json')
        if not self.metadata and self.metadata_path is not None:
            self.metadata = _read_json(self.metadata_path)
        if self.array is None and self.path is None:
            raise ValueError('A map requires an array or an NPY path')

    def load(self):
        """Use the existing array directly, otherwise map the NPY read-only."""
        return self.array if self.array is not None else np.load(
            self.path, mmap_mode='r', allow_pickle=False)


class MagnificationMapSeries:
    """Read a manifest without loading any map arrays."""
    def __init__(self, manifest):
        path = Path(manifest).expanduser().resolve()
        self.manifest_path = path / 'manifest.json' if path.is_dir() else path
        record = _read_json(self.manifest_path)
        if record.get('format_version') != 1 or not record.get('frames'):
            raise ValueError('Expected a nonempty DCLens format_version=1 manifest')
        self.frames = record['frames']
        self.geometry = record['geometry']
        self.simulation = record.get('simulation', {})
        for i, frame in enumerate(self.frames):
            if frame['index'] != i:
                raise ValueError('Manifest frame indices must be contiguous from zero')

    @property
    def frame_count(self):
        return len(self.frames)

    def frame(self, i):
        record = self.frames[i]
        parent = self.manifest_path.parent
        return MagnificationMap(path=parent / record['map_file'],
                                metadata_path=parent / record['metadata_file'])

    def validate(self):
        """Check every header and metadata record without reading map pixels."""
        for i in range(self.frame_count):
            frame = self.frame(i)
            array = frame.load()
            geometry = _map_geometry(frame, array)
            if any(geometry[key] != self.geometry.get(key) for key in _GEOMETRY_KEYS):
                raise ValueError(f'Incompatible map geometry at frame {i}')
            del array


def _as_map(source, frame=0):
    if isinstance(source, MagnificationMapSeries):
        return source.frame(frame)
    if isinstance(source, MagnificationMap):
        return source
    if isinstance(source, np.ndarray):
        return MagnificationMap(array=source)
    path = Path(source)
    if path.is_dir() or path.suffix == '.json':
        return MagnificationMapSeries(path).frame(frame)
    return MagnificationMap(path=path)


def _map_geometry(mmp, array, ps=None, img_center=(0., 0.)):
    if array.ndim != 2 or min(array.shape) < 1 or array.dtype.kind not in 'fiu':
        raise ValueError('A magnification map must be a nonempty real 2-D array')
    ny, nx = array.shape
    if 'geometry' in mmp.metadata:
        geometry = dict(mmp.metadata['geometry'])
        if geometry['nx'] != nx or geometry['ny'] != ny:
            raise ValueError('Map shape disagrees with metadata')
        if ps is not None and ps != geometry['srcp_ps']:
            raise ValueError('Explicit ps disagrees with metadata')
    else:
        if ps is None or not np.isfinite(ps) or ps <= 0:
            raise ValueError('Raw arrays/NPY files without metadata require positive ps')
        cx, cy = img_center
        geometry = dict(src_xmin=cx-nx*ps/2, src_xmax=cx+nx*ps/2,
                        src_ymin=cy-ny*ps/2, src_ymax=cy+ny*ps/2,
                        srcp_ps=float(ps), nx=nx, ny=ny,
                        coordinate_convention=_COORDINATES)
    scale = geometry['srcp_ps']
    if not np.isfinite([geometry[k] for k in _GEOMETRY_KEYS[:5]]).all() or scale <= 0:
        raise ValueError('Non-finite or nonpositive map geometry')
    if geometry.get('coordinate_convention') != _COORDINATES:
        raise ValueError('Unsupported map coordinate convention')
    for axis, size in [('x', nx), ('y', ny)]:
        span = geometry[f'src_{axis}max'] - geometry[f'src_{axis}min']
        if not np.isclose(span, size*scale, rtol=1e-12, atol=scale*1e-10):
            raise ValueError('Map bounds disagree with pixel scale')
    return geometry
