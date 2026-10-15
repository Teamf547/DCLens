"""Continuous source paths sampled by one streaming CUDA convolution backend."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import DAYS_PER_YEAR
from .magnification_map import (MagnificationMapSeries, _as_map, _map_geometry)


@dataclass
class LightCurveResult:
    time_days: np.ndarray
    magnification: np.ndarray
    x: np.ndarray
    y: np.ndarray
    frame_index: np.ndarray

    def save(self, path):
        """Save named arrays without pickle, with time explicitly in days."""
        path = Path(path).expanduser()
        if path.suffix != '.npz':
            path = Path(str(path) + '.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, time_days=self.time_days, magnification=self.magnification,
                 x=self.x, y=self.y, frame_index=self.frame_index)
        return path


def _source_kernel(radius):
    if not np.isfinite(radius):
        raise ValueError('source_radius_px must be finite')
    # Preserve the original truncated Gaussian square: r=int(radius)>=1,
    # sigma=r/2, support [-r,r]^2, and discrete total flux exactly normalized.
    r = max(1, int(radius))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    kernel = np.exp(-(x*x+y*y)/(2*(r/2.)**2))
    return kernel/kernel.sum(), r


def _roi_bounds(rows, cols, radius):
    return (int(rows.min())-radius, int(rows.max())+radius+1,
            int(cols.min())-radius, int(cols.max())+radius+1)


class IALightCurveExtractor:
    """GPU extraction from a map, series, manifest, or maps directory.

    Construction performs no plotting or map upload. Call extract() in scripts
    or show() for the two-click interactive view. Saved arrays use read-only
    mmap; only the current path ROI is made contiguous and uploaded.
    """
    def __init__(self, source, ps=None, *, frame_dt_days=None,
                 pmv_arcsec_yr=.005, max_vram_mb=2048, img_center=(0., 0.),
                 title_mmp='Magnification Map', lc_title='Light Curve', gamma=.1):
        self.series = None
        if isinstance(source, MagnificationMapSeries):
            self.series = source
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir() or path.suffix == '.json':
                self.series = MagnificationMapSeries(path)
        if not np.isfinite(pmv_arcsec_yr) or pmv_arcsec_yr <= 0:
            raise ValueError('pmv_arcsec_yr must be finite and positive')
        if max_vram_mb is not None and (not np.isfinite(max_vram_mb) or max_vram_mb <= 0):
            raise ValueError('max_vram_mb must be positive or None')
        if self.series is not None:
            if frame_dt_days is None or not np.isfinite(frame_dt_days) or frame_dt_days <= 0:
                raise ValueError('A map sequence requires positive frame_dt_days')
            self.map = None
            initial = self.series.frame(0)
        else:
            self.map = _as_map(source)
            initial = self.map
        array = initial.load()
        self.geometry = _map_geometry(initial, array, ps, img_center)
        del array
        self.frame_dt_days = frame_dt_days
        self.pmv_arcsec_yr = float(pmv_arcsec_yr)
        self.max_vram_mb = max_vram_mb
        self.img_center = img_center
        self.title_mmp, self.lc_title, self.gamma = title_mmp, lc_title, gamma
        self._view = None

    def _trajectory(self, start, end, sample_dt_days):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if start.shape != (2,) or end.shape != (2,) or not np.isfinite([start,end]).all():
            raise ValueError('start and end must be finite (x, y) arcsecond pairs')
        distance = float(np.linalg.norm(end-start))
        duration = distance/self.pmv_arcsec_yr*DAYS_PER_YEAR
        map_duration = (self.series.frame_count*self.frame_dt_days
                        if self.series is not None else np.inf)
        finish = min(duration, map_duration)
        if sample_dt_days is not None and (not np.isfinite(sample_dt_days) or sample_dt_days <= 0):
            raise ValueError('sample_dt_days must be positive')
        if duration == 0:
            times = np.array([0.])
        elif sample_dt_days is None:
            # Keep the established density (roughly one point per map pixel,
            # at least ten per full path), with physical angular travel time.
            count = int(max(distance/self.geometry['srcp_ps'], 10))
            interval = duration/(count-1)
            times = np.arange(min(count, int(np.floor(finish/interval))+1))*interval
            if duration < map_duration and times[-1] != duration:
                times = np.append(times[times<duration], duration)
        else:
            times = np.arange(int(np.floor(finish/sample_dt_days))+1)*sample_dt_days
            if duration < map_duration and times[-1] < duration:
                times = np.append(times, duration)
        # Frames cover half-open intervals. The final map's right edge has no
        # snapshot; never repeat the last map or extrapolate beyond the series.
        times = times[(times <= duration) & (times < map_duration)]
        fractions = times/duration if duration else np.zeros_like(times)
        positions = start[None,:] + fractions[:,None]*(end-start)[None,:]
        if self.series is None:
            frames = np.zeros(times.size, dtype=np.int64)
        else:
            # Equivalent to floor(t/dt), comparing actual interval boundaries
            # also avoids 0.3/0.1 rounding an exact generated boundary downward.
            edges = np.arange(self.series.frame_count+1)*self.frame_dt_days
            frames = np.searchsorted(edges, times, side='right')-1
        return times, positions[:,0], positions[:,1], frames

    def extract(self, start, end, source_radius_px=5, *, sample_dt_days=None):
        """Return time, magnification, continuous x/y and snapshot indices.

        The source window is centered on the integer pixel (no interpolation).
        Windows outside the map retain the original magnification=1 boundary.
        max_vram_mb bounds all reusable extraction buffers, not just ROI data.
        """
        if self.series is not None:
            self.series.validate()  # All metadata/headers checked before GPU work.
        time, x, y, frames = self._trajectory(start, end, sample_dt_days)
        kernel, radius = _source_kernel(source_radius_px)
        geo = self.geometry
        # int() truncation matches the previous world_to_pixel convention.
        rows = np.trunc((y-geo['src_ymin'])/geo['srcp_ps']).astype(np.int64)
        cols = np.trunc((x-geo['src_xmin'])/geo['srcp_ps']).astype(np.int64)
        magnification = np.ones(time.size, dtype=np.float64)
        valid = ((rows>=radius) & (rows<geo['ny']-radius) &
                 (cols>=radius) & (cols<geo['nx']-radius))
        if not valid.any():
            return LightCurveResult(time, magnification, x, y, frames)
        from . import _cuda
        free, _ = _cuda.memory_info()
        requested = (2048 if self.max_vram_mb is None else self.max_vram_mb)*1024**2
        # Current free VRAM accounts for desktop, CUDA and OpenGL contexts.
        # Use at most half of it and then leave another 20% workspace headroom.
        limit = int(min(requested, free*.5)*.8)
        minimum = kernel.size*4 + kernel.nbytes + 16
        if minimum > limit:
            raise RuntimeError('Insufficient GPU memory for light-curve extraction: one source window exceeds budget')
        gpu = None
        try:
            try:
                gpu = _cuda.LightCurveWorkspace(kernel)
            except _cuda.CudaMemoryError as error:
                raise RuntimeError('Insufficient GPU memory for light-curve source kernel') from error
            for frame in np.unique(frames[valid]):
                indices = np.flatnonzero((frames==frame) & valid)
                mmp = self.series.frame(int(frame)) if self.series is not None else self.map
                array = mmp.load()
                if _map_geometry(mmp, array, geo['srcp_ps'], self.img_center) != geo:
                    raise ValueError(f'Map geometry changed at frame {frame}')
                offset = 0
                max_batch_size = indices.size
                while offset < indices.size:
                    batch_size = min(max_batch_size, indices.size-offset)
                    while True:
                        selected = indices[offset:offset+batch_size]
                        rr, cc = rows[selected], cols[selected]
                        r0, r1, c0, c1 = _roi_bounds(rr, cc, radius)
                        required = (r1-r0)*(c1-c0)*4 + batch_size*16 + kernel.nbytes
                        if required > limit:
                            if batch_size == 1:
                                raise RuntimeError('Insufficient GPU memory for light-curve extraction: minimum batch cannot fit')
                            batch_size = max(1,batch_size//2)
                            continue
                        roi = np.ascontiguousarray(array[r0:r1,c0:c1], dtype=np.float32)
                        try:
                            magnification[selected] = gpu.extract(
                                roi, np.asarray(rr-r0,dtype=np.int32),
                                np.asarray(cc-c0,dtype=np.int32), limit)
                        except _cuda.CudaMemoryError as error:
                            # One explicit recovery policy, no alternate backend.
                            gpu.clear_batches()
                            if batch_size == 1:
                                raise RuntimeError('Insufficient GPU memory for light-curve extraction at minimum batch size') from error
                            batch_size = max(1,batch_size//2)
                        else:
                            break
                        finally:
                            del roi
                    offset += batch_size
                    max_batch_size = batch_size
                del array, mmp
        finally:
            if gpu is not None:
                gpu.close()
        return LightCurveResult(time, magnification, x, y, frames)

    def show(self):
        """Show frame zero with two-click path selection and a radius slider."""
        from .visualization import _LightCurveView
        if self._view is None:
            self._view = _LightCurveView(self)
        return self._view.show()
