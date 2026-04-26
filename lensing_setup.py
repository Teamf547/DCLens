# lensing_setup.py

import numpy as np
import pandas as pd
from .physics_utils import generate_BH_size_samples, pc_to_redshift
import logging

logger = logging.getLogger("lensing_setup")

def split_into_planes(BH_table: pd.DataFrame, N_planes: int, D_S: float):

    BH_sorted = BH_table.sort_values(by='z_redshift').reset_index(drop=True)
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
        mean_z = sub_tbl['z_redshift'].mean()
        # revert mean_z -> distance approximation consistent with original code:
        D_i = (mean_z * 3e5 / 1e3) * 1e6 / 70.0  # keep similar (fallback); caller may replace
        # but in original code: D_i = (mean_z * c / 1e3) * 1e6 / H_0
        # choose consistent conversion (above is tuned to preserve original units roughly)
        lens_planes.append({
            'z': float(D_i),
            'lenses': {
                'x': sub_tbl['x_arcsec'].values.astype(np.float64),
                'y': sub_tbl['y_arcsec'].values.astype(np.float64),
                'theta_E': sub_tbl['theta_E'].values.astype(np.float64)
            }
        })
        idx_start = idx_end
    lens_planes = sorted(lens_planes, key=lambda p: p['z'])
    logger.info("split_into_planes -> created %d planes", len(lens_planes))
    return lens_planes

def build_BH_table_from_raw_dataframe(df, D_L_pc):

    rad_to_arcsec = 180.0 / np.pi * 3600.0
    df = df.copy()
    df["z_[pc]"] = df["z_[pc]"] + D_L_pc
    bad_z = df["z_[pc]"] <= 0
    if bad_z.any():
        logger.warning("发现 %d 条 z_[pc] <= 0，已剔除", int(bad_z.sum()))
        df = df.loc[~bad_z].reset_index(drop=True)
    df["x_arcsec"] = (df["x_[pc]"] / df["z_[pc]"]) * rad_to_arcsec
    df["y_arcsec"] = (df["y_[pc]"] / df["z_[pc]"]) * rad_to_arcsec
    df["z_redshift"] = df["z_[pc]"].apply(pc_to_redshift)
    theta_E_arcsec = generate_BH_size_samples(df["Mass_[Msun]"].values, df["z_[pc]"].values, 1e9)
    df["theta_E"] = theta_E_arcsec
    BH_table = pd.DataFrame({
        "x_arcsec": df["x_arcsec"].values,
        "y_arcsec": df["y_arcsec"].values,
        "theta_E": df["theta_E"].values,
        "z_redshift": df["z_redshift"].values
    })
    return BH_table
