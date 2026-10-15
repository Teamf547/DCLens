"""Prepare lens-plane and P3M data for the CUDA lensing engine.

The module splits a physical lens catalogue into distance-ordered planes, then
builds the PM mesh and PP buckets consumed by ``_cuda``. Lens positions,
plane distances, mesh origins, and mesh spacing are always in parsecs; masses
are in solar masses. Image-plane bounds are accepted in arcseconds and are
converted only to determine the ray footprint on each physical lens plane.

The numerical P3M choices are fixed: TSC assignment (p=2), ``pp_h == a_pm``,
and a 3x3 PP bucket search in CUDA. This module exposes only the supported mesh
resolution, ``a_pm/H``, and padding controls.
"""

import logging
logger = logging.getLogger("DCLens")

import numpy as np
import pandas as pd

def split_into_planes(BH_table: pd.DataFrame, N_planes: int):
    """Split a lens table into distance-ordered, nearly equal-sized planes."""

    BH_sorted = BH_table.sort_values(by='z_pc').reset_index(drop=True)
    N_planes = min(N_planes, len(BH_sorted))
    if N_planes <= 0:
        return []
    counts = np.full(N_planes, len(BH_sorted) // N_planes, dtype=int)
    counts[:len(BH_sorted) % N_planes] += 1
    lens_planes = []
    idx_start = 0
    for i in range(N_planes):
        cnt = counts[i]
        idx_end = idx_start + cnt
        sub_tbl = BH_sorted.iloc[idx_start:idx_end]
        if len(sub_tbl) == 0:
            idx_start = idx_end
            continue
        D_i_pc = sub_tbl['z_pc'].mean()

        lens_planes.append({
            'z_pc': float(D_i_pc),
            'lenses': {
                'x_pc': sub_tbl['x_pc'].values.astype(np.float64),
                'y_pc': sub_tbl['y_pc'].values.astype(np.float64),
                'M_Msun': sub_tbl['M_Msun'].values.astype(np.float64)
            }
        })
        idx_start = idx_end
    lens_planes = sorted(lens_planes, key=lambda p: p['z_pc'])
    logger.debug("[Input] Split lenses into %d plane(s)", len(lens_planes))
    return lens_planes

def _next_pow2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 2  # minimum FFT size
    return 1 << (int(n - 1).bit_length())


def apply_p3m_grid(lens_planes, grid_nx=128, grid_ny=128,
                   theta_bounds_arcsec=None,
                   a_pm_over_H=6.0,
                   padding_factor=2.0):
    """Build the PM meshes and PP buckets for every lens plane.

    A single square spacing ``H`` is chosen from the requested active-grid
    resolution. The padded periodic FFT domain stores ``pm_nx * pm_ny`` unique
    samples; it does not duplicate a periodic endpoint. PP buckets have width
    ``a_pm``, so every lens inside the cutoff is in the current bucket or one
    of its eight neighbours.

    Parameters
    ----------
    lens_planes : list of dict
        split_into_planes() 的输出
    grid_nx, grid_ny : int
        活跃区域内期望的 PM mesh 分辨率 (未 padding 前)
    theta_bounds_arcsec : tuple or None
        (theta_xmin, theta_xmax, theta_ymin, theta_ymax) arcsec
        Image-plane ray footprint used to include propagated rays in the mesh.
    a_pm_over_H : float
        Supported P3M splitting-radius/mesh-spacing ratio (default 6.0).
    padding_factor : float
        FFT-box padding factor (default 2.0).

    Internal units
    --------------
    Lens.x/y : pc,  Lens.M : Msun,  plane.z_pc : pc
    PM mesh coordinates and spacing H : pc
    """
    ARCSEC2RAD = 4.84813681109536e-6

    if grid_nx <= 0 or grid_ny <= 0:
        raise ValueError("grid_nx and grid_ny must be positive")
    if a_pm_over_H <= 0.0:
        raise ValueError("a_pm_over_H must be positive")
    if padding_factor <= 0.0:
        raise ValueError("padding_factor must be positive")

    for plane_index, plane in enumerate(lens_planes, start=1):
        x = plane['lenses']['x_pc']
        y = plane['lenses']['y_pc']
        M = plane['lenses']['M_Msun']

        if len(x) == 0:
            # Keep a minimal, type-compatible layout for an empty plane.
            plane['p3m'] = {
                'pm_nx': 2, 'pm_ny': 2,
                'pm_xmin': 0.0, 'pm_ymin': 0.0,
                'H': 1.0, 'a_pm': float(a_pm_over_H),
                'pp_nx': 1, 'pp_ny': 1,
                'pp_xmin': 0.0, 'pp_ymin': 0.0,
                'pp_h': float(a_pm_over_H),
                'raw_nx': 2, 'raw_ny': 2,
                'active_grid_nx': int(grid_nx),
                'active_grid_ny': int(grid_ny),
                'bucket_start': np.zeros(1, dtype=np.int32),
                'bucket_count': np.zeros(1, dtype=np.int32),
            }
            continue

        z = plane['z_pc']

        # --- 6.1 确定 active domain ---
        lens_xmin, lens_xmax = float(x.min()), float(x.max())
        lens_ymin, lens_ymax = float(y.min()), float(y.max())

        active_xmin = lens_xmin
        active_xmax = lens_xmax
        active_ymin = lens_ymin
        active_ymax = lens_ymax

        if theta_bounds_arcsec is not None:
            t_xmin, t_xmax, t_ymin, t_ymax = theta_bounds_arcsec
            # 将 image-plane arcsec 坐标转换为该 plane 的 pc 坐标
            ray_xmin_pc = t_xmin * ARCSEC2RAD * z
            ray_xmax_pc = t_xmax * ARCSEC2RAD * z
            ray_ymin_pc = t_ymin * ARCSEC2RAD * z
            ray_ymax_pc = t_ymax * ARCSEC2RAD * z
            active_xmin = min(active_xmin, ray_xmin_pc)
            active_xmax = max(active_xmax, ray_xmax_pc)
            active_ymin = min(active_ymin, ray_ymin_pc)
            active_ymax = max(active_ymax, ray_ymax_pc)

        span_x = active_xmax - active_xmin
        span_y = active_ymax - active_ymin

        # 防止退化 (所有 lens 在同一点)
        if span_x < 1e-10:
            span_x = 1e-5
            active_xmin -= span_x / 2
            active_xmax += span_x / 2
        if span_y < 1e-10:
            span_y = 1e-5
            active_ymin -= span_y / 2
            active_ymax += span_y / 2

        # --- 6.1 统一正方形 mesh spacing H ---
        H_x = span_x / grid_nx
        H_y = span_y / grid_ny
        H = max(H_x, H_y)

        # --- 6.2 P3M cutoff ---
        a_pm = a_pm_over_H * H

        # --- 6.3 PM FFT box padding ---
        cx = 0.5 * (active_xmin + active_xmax)
        cy = 0.5 * (active_ymin + active_ymax)

        padded_span_x = max(span_x * padding_factor,
                            span_x + 2.0 * (a_pm + 2 * H))
        padded_span_y = max(span_y * padding_factor,
                            span_y + 2.0 * (a_pm + 2 * H))

        # A periodic FFT mesh stores no repeated endpoint. Avoid turning an
        # exact power-of-two requirement (for example 4096) into 4097/8192.
        raw_nx = max(2, int(np.ceil(padded_span_x / H)))
        raw_ny = max(2, int(np.ceil(padded_span_y / H)))

        pm_nx = _next_pow2(raw_nx)
        pm_ny = _next_pow2(raw_ny)

        if pm_nx * H < padded_span_x or pm_ny * H < padded_span_y:
            raise RuntimeError("FFT mesh does not cover the padded physical domain")

        # PM grid origin (H 不变，根据最终 pm_nx/pm_ny 重新确定 origin)
        pm_xmin = cx - 0.5 * (pm_nx - 1) * H
        pm_ymin = cy - 0.5 * (pm_ny - 1) * H

        # --- 7. 构建独立 PP bucket ---
        pp_h = a_pm  # H_bucket = a_pm

        pp_xmin = np.floor((lens_xmin - a_pm) / pp_h) * pp_h
        pp_ymin = np.floor((lens_ymin - a_pm) / pp_h) * pp_h
        pp_xmax = np.ceil((lens_xmax + a_pm) / pp_h) * pp_h
        pp_ymax = np.ceil((lens_ymax + a_pm) / pp_h) * pp_h

        pp_nx = max(1, int(np.ceil((pp_xmax - pp_xmin) / pp_h)))
        pp_ny = max(1, int(np.ceil((pp_ymax - pp_ymin) / pp_h)))

        # 计算每个 lens 的 bucket id
        bix = np.floor((x - pp_xmin) / pp_h).astype(np.int64)
        biy = np.floor((y - pp_ymin) / pp_h).astype(np.int64)
        bix = np.clip(bix, 0, pp_nx - 1)
        biy = np.clip(biy, 0, pp_ny - 1)
        bucket_id = biy * pp_nx + bix

        # 按 bucket_id 排序 lens (保证同 bucket 内 lens 连续)
        order = np.argsort(bucket_id, kind='stable')
        plane['lenses']['x_pc'] = x[order]
        plane['lenses']['y_pc'] = y[order]
        plane['lenses']['M_Msun'] = M[order]

        bucket_id_sorted = bucket_id[order]

        num_buckets = pp_nx * pp_ny
        bucket_count = np.bincount(
            bucket_id_sorted, minlength=num_buckets
        ).astype(np.int32)

        # Accumulate in int64, but preserve the C++ ``std::vector<int>`` ABI.
        bucket_start = np.zeros(num_buckets, dtype=np.int32)
        if num_buckets > 1:
            cs = np.cumsum(bucket_count[:-1], dtype=np.int64)
            if cs[-1] > np.iinfo(np.int32).max:
                raise OverflowError(
                    "PP bucket prefix exceeds int32 capacity required by _cuda"
                )
            bucket_start[1:] = cs.astype(np.int32)

        # 日志
        mean_lenses_per_bucket = len(x) / max(1, num_buckets)
        max_lenses_per_bucket = int(bucket_count.max()) if len(bucket_count) > 0 else 0
        logger.debug(
            "[P3M] Plane %d/%d: %d lenses | active %d x %d | raw FFT %d x %d | "
            "FFT %d x %d | H=%.4e pc | a_pm=%.4e pc (a_pm/H=%.3f) | "
            "PP %d x %d (mean %.1f, max %d lenses/bucket)",
            plane_index, len(lens_planes), len(x), grid_nx, grid_ny,
            raw_nx, raw_ny, pm_nx, pm_ny, H, a_pm, a_pm_over_H,
            pp_nx, pp_ny, mean_lenses_per_bucket, max_lenses_per_bucket
        )

        # --- 8. 保存 P3M 数据 ---
        plane['p3m'] = {
            # PM mesh
            'pm_nx': int(pm_nx),
            'pm_ny': int(pm_ny),
            'pm_xmin': float(pm_xmin),
            'pm_ymin': float(pm_ymin),
            'H': float(H),
            'a_pm': float(a_pm),
            'raw_nx': int(raw_nx),
            'raw_ny': int(raw_ny),
            'active_grid_nx': int(grid_nx),
            'active_grid_ny': int(grid_ny),
            # PP buckets
            'pp_nx': int(pp_nx),
            'pp_ny': int(pp_ny),
            'pp_xmin': float(pp_xmin),
            'pp_ymin': float(pp_ymin),
            'pp_h': float(pp_h),
            'bucket_start': bucket_start,
            'bucket_count': bucket_count,
        }

    return lens_planes


def _pack_p3m_params(p3m):
    """Pack one plane in the positional order parsed by ``load_planes``."""
    return [
        p3m["pm_nx"], p3m["pm_ny"], p3m["pm_xmin"], p3m["pm_ymin"],
        p3m["H"], p3m["a_pm"],
        p3m["pp_nx"], p3m["pp_ny"], p3m["pp_xmin"], p3m["pp_ymin"],
        p3m["pp_h"],
    ]

def _prepare_engine_planes(lens_planes, D_S_pc):
    """Preserve plane order, dtypes, and units while packing pybind inputs."""
    lx_list, ly_list, mass_list = [], [], []
    bucket_start_list, bucket_count_list, p3m_params = [], [], []
    z_list, factor_list = [], []

    for plane in lens_planes:
        z_pc = plane["z_pc"]
        p3m = plane["p3m"]
        z_list.append(z_pc)
        factor_list.append((D_S_pc - z_pc) / D_S_pc)
        lx_list.append(plane["lenses"]["x_pc"])
        ly_list.append(plane["lenses"]["y_pc"])
        mass_list.append(plane["lenses"]["M_Msun"])
        bucket_start_list.append(p3m["bucket_start"])
        bucket_count_list.append(p3m["bucket_count"])
        p3m_params.append(_pack_p3m_params(p3m))

    return (
        lx_list, ly_list, mass_list,
        bucket_start_list, bucket_count_list, p3m_params,
        z_list, factor_list,
    )

def _p3m_plane_metadata(lens_planes):
    """Return JSON-serializable P3M geometry for reproducible runs."""
    result = []
    for plane in lens_planes:
        p3m = plane["p3m"]
        result.append({
            "z_pc": float(plane["z_pc"]),
            "lens_count": int(len(plane["lenses"]["x_pc"])),
            "H_pc": float(p3m["H"]),
            "a_pm_pc": float(p3m["a_pm"]),
            "pm_nx": int(p3m["pm_nx"]),
            "pm_ny": int(p3m["pm_ny"]),
            "pm_xmin_pc": float(p3m["pm_xmin"]),
            "pm_ymin_pc": float(p3m["pm_ymin"]),
            "pp_nx": int(p3m["pp_nx"]),
            "pp_ny": int(p3m["pp_ny"]),
            "pp_h_pc": float(p3m["pp_h"]),
        })
    return result


# Diagram-facing name for the existing host mesh builder; no wrapper or copy.
PMBuilder = apply_p3m_grid

__all__ = ['PMBuilder', 'apply_p3m_grid', 'split_into_planes']
