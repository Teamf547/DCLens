# MMpS.py -- Magnification Map Simulation and Showing
import os
import sys
import json
from .logger_config import setup_logger
logger = setup_logger("DCLens")
import numpy as np
import pandas as pd
import math

from .constants import pc_in_m
from .physics_utils import generate_BH_size_samples, pc_to_redshift
from .lensing_setup import split_into_planes
from .gpu_rasterizer import rasterize_triangles_to_texture_tiled

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FuncFormatter

# import matplotlib
# matplotlib.use('agg')

import cv2

import time

from . import lensing_core

# Magnification Map Simulation
def MagMapSim(ini_con_file,
            D_L_pc=1e4, D_S_pc=1e9,
            N_planes=1, coarse_ps=1e-3, finest_ps=1e-4,
            img_center=(0.0, 0.0), src_considered_region=(0.1, 0.1), buffer_size=0.1,
            max_refine_iters=20,
            srcp_ps=1e-4,
            ss=1, max_tile_dim=40960, batch_triangles=200000,
            output_npy="mmp.npy", save_dir='.',
            ):
    
    start_time = time.time()
    ## Step1: load and process lens plane data
    logger.info("loading and processing lens plane data...")
    df = pd.read_csv(
        ini_con_file, sep=r"\s+", header=None, comment='#',
        names=["Mass_[Msun]", "x_[pc]", "y_[pc]", "z_[pc]", "vx_[km/s]", "vy_[km/s]", "vz_[km/s]"]
    )
    df["z_[pc]"] = df["z_[pc]"] + D_L_pc
    df = df.loc[df["z_[pc]"] > 0].reset_index(drop=True)

    BH_table = pd.DataFrame({
        "x_arcsec": (df["x_[pc]"] / df["z_[pc]"]) * (180.0 / math.pi * 3600.0), # transform pc to arsec
        "y_arcsec": (df["y_[pc]"] / df["z_[pc]"]) * (180.0 / math.pi * 3600.0),
        "theta_E": generate_BH_size_samples(df["Mass_[Msun]"].values, df["z_[pc]"].values, D_S_pc), # calculate einstein radius
        "z_redshift": df["z_[pc]"].apply(pc_to_redshift)
    })
    
    lens_planes = split_into_planes(BH_table, N_planes=N_planes, D_S=D_S_pc)
    # total_lenses = sum(len(p['lenses']['x']) for p in lens_planes)
    total_lenses = BH_table.shape[0]
    logger.info(f"total lenses number: {total_lenses}, distribute on {len(lens_planes)} planes.")
    

    ## step2: initialize GPU engine
    engine = lensing_core.LensingEngineGPU(D_S_pc, 20000000, 20000000) #giving large enough capacity for queue avoiding refine's break
    logger.info("transfer lenses data to GPU...")
    
    # prepare data struct
    lx_list, ly_list, lE_list = [], [], []
    z_list, factor_list = [], []

    for plane in lens_planes:
        z_list.append(plane['z'])
        D_iS = D_S_pc - plane['z']
        factor_list.append(D_iS / D_S_pc) # the projection factor
        lx_list.append(plane['lenses']['x'])
        ly_list.append(plane['lenses']['y'])
        lE_list.append(plane['lenses']['theta_E'])
    
    # upload to GPU
    engine.load_planes(lx_list, ly_list, lE_list, z_list, factor_list)


    ## step3: determine the region
    xc, yc = img_center
    w, h = src_considered_region
    consider_xmin = xc - w/2
    consider_xmax = xc + w/2
    consider_ymin = yc - h/2
    consider_ymax = yc + h/2
    
    img_xmin = consider_xmin - buffer_size
    img_xmax = consider_xmax + buffer_size
    img_ymin = consider_ymin - buffer_size
    img_ymax = consider_ymax + buffer_size

    ## step4: run GPU adaptive quadtree refinement
    logger.info("running the GPU adaptive quadtree refinement geometic ray-shooting...")
    
    cpp_start = time.time()
    mu_upper_val = 4.0
    mu_lower_val = 0.6
    results = engine.run_adaptive(
        img_xmin, img_xmax, img_ymin, img_ymax,
        coarse_ps, finest_ps,
        max_refine_iters,
        mu_upper_val, mu_lower_val,
        consider_xmin, consider_xmax, consider_ymin, consider_ymax
    )
    cpp_end = time.time()
    
    mapped_pts_all = results["mapped_pts"]    # shape (N, 4, 2), the corners of geometric ray elements
    mags_all = results["magnifications"]      # shape (N,)
    pix_sizes_all = results["pixel_sizes"]    # shape (N,)
    
    total_leaves = len(mags_all)
    logger.info("细分完成，获得叶子节点: %d", total_leaves)

    ## step5: triangularization
    nx_src = int(np.ceil(w / srcp_ps))
    ny_src = int(np.ceil(h / srcp_ps))
    src_xmax = consider_xmin + nx_src * srcp_ps
    src_ymax = consider_ymin + ny_src * srcp_ps
    
    logger.info("Vectorized triangularizating...")
    
    valid_mask = np.isfinite(mags_all) & (mags_all > 0)
    valid_pts = mapped_pts_all[valid_mask] # shape (M, 4, 2)
    valid_mags = mags_all[valid_mask]      # shape (M,)
    
    # normlization coordinates
    u = (valid_pts[:, :, 0] - consider_xmin) / (src_xmax - consider_xmin)
    v = (valid_pts[:, :, 1] - consider_ymin) / (src_ymax - consider_ymin)
    
    # T1: 0-1-2
    t1_u = np.stack([u[:,0], u[:,1], u[:,2]], axis=1) # (M, 3)
    t1_v = np.stack([v[:,0], v[:,1], v[:,2]], axis=1)
    
    # T2: 0-2-3
    t2_u = np.stack([u[:,0], u[:,2], u[:,3]], axis=1)
    t2_v = np.stack([v[:,0], v[:,2], v[:,3]], axis=1)
    
    # concatenate
    all_u = np.concatenate([t1_u, t2_u], axis=0) # (2M, 3)
    all_v = np.concatenate([t1_v, t2_v], axis=0)
    all_mags = np.tile(valid_mags, 2) # (2M,)
    
    # flatten (6M, 3) -> [u, v, val]
    n_tris = all_u.shape[0]
    verts = np.zeros((n_tris * 3, 3), dtype=np.float32)
    
    # fill U, V
    verts[:, 0] = all_u.reshape(-1)
    verts[:, 1] = all_v.reshape(-1)
    
    # mag repeated for each vertex
    val_repeated = np.repeat(all_mags, 3)
    # verts[:, 2] = val_repeated
    verts[:, 2] = np.ones_like(val_repeated)
    
    logger.info("顶点构建完成: %d verts. 开始 GPU 光栅化...", verts.shape[0])

    # step6. GPU shader rasterization
    pixel_mu_map = rasterize_triangles_to_texture_tiled(
        verts, nx_src, ny_src, ss=ss, 
        max_tile_dim=max_tile_dim, batch_triangles=batch_triangles
    )
    
    # step7. save results and metadata
    normli = np.mean(pixel_mu_map)
    pixel_mu_map = pixel_mu_map / normli if normli > 0 else pixel_mu_map
    
    os.makedirs(save_dir, exist_ok=True)
    outpath = os.path.join(save_dir, output_npy)
    np.save(outpath, pixel_mu_map)
    
    # json metadata
    end_time = time.time()
    meta = {
        'run_info': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_duration_sec': round(end_time - start_time, 2),
            'cpp_refinement_duration_sec': round(cpp_end - cpp_start, 2),
            'total_leaf_nodes': len(mags_all),
            'total_lenses_processed': total_lenses
        },
        'src_grid': {
            'src_xmin': consider_xmin, 
            'src_xmax': src_xmax,
            'src_ymin': consider_ymin,
            'src_ymax': src_ymax,
            'srcp_ps': srcp_ps,
            'nx_src': nx_src,
            'ny_src': ny_src
        },
        'parameters': {
            'coarse_ps': coarse_ps,
            'finest_ps': finest_ps,
            'max_refine_iters': max_refine_iters,
            'mu_upper_thresholds': mu_upper_val,
            'mu_lower_threshold': mu_lower_val,
            'D_L_pc': D_L_pc,
            'D_S_pc': D_S_pc,
            'N_planes': len(lens_planes),
            'supersampling_ss': ss,
            'buffer_size': buffer_size
        },
        'image_plane_region': {
            'img_center': img_center,
            'img_xmin': img_xmin,
            'img_xmax': img_xmax,
            'img_ymin': img_ymin,
            'img_ymax': img_ymax
        }
    }

    with open(outpath + ".meta.json", 'w') as fh:
        json.dump(meta, fh, indent=4)
    
    logger.info(f"详细运行参数已记录至 {outpath}.meta.json")
    return {'pixel_mu_map': pixel_mu_map}

from IPython.display import display, clear_output

def MagMapShow(mmp_file, 
               ps, cut=0, 
               gamma=0.1,
               sys_name='N-body Lens System',
               patch_mode=False,
               n_patch:int=2):
    
    H = np.load(mmp_file)
    H0 = H[cut:H.shape[0]-cut, cut:H.shape[1]-cut]
    Hi = []
    if patch_mode:
        patch_size_x = H0.shape[1] // n_patch
        patch_size_y = H0.shape[0] // n_patch
        for i in range(n_patch):
            for j in range(n_patch):
                patch = H0[i*patch_size_y:(i+1)*patch_size_y, j*patch_size_x:(j+1)*patch_size_x]
                Hi.append(patch)
    else:
        n_patch = 1
        Hi.append(H0)

    x_len = H0.shape[0] // n_patch
    y_len = H0.shape[1] // n_patch
    x1_min = -x_len * ps / 2
    x1_max = x_len * ps / 2
    y1_min = -y_len * ps / 2
    y1_max = y_len * ps / 2

    for idx, hi in enumerate(Hi):
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        
        vmin1, vmax1 = hi.min(), hi.max()
        norm1 = PowerNorm(gamma=gamma, vmin=vmin1, vmax=vmax1)

        im1 = axs.imshow(hi, origin='lower',
                        cmap='magma',
                        extent=[x1_min, x1_max, y1_max, y1_min],
                        aspect='auto', norm=norm1)

        axs.set_title(f'Magnification Map of SRC plane({sys_name}) - Patch {idx+1}/{len(Hi)}')
        axs.set_xlabel("βₓ (arcsec)")
        axs.set_ylabel("βᵧ (arcsec)")
        axs.set_aspect('equal', adjustable='box')

        cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.046, pad=0.02)
        cbar.set_label('Magnification')
        cbar_ax = fig.add_axes([0.13, 0.03, 0.765, 0.03])  # [left, bottom, width, height]
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        custom_ticks = [0.5, 1, 10, 100, 500]
        cbar.set_ticks(custom_ticks)
        def custom_fmt(x, pos):
            if x >= 100:
                return f"${x/(10**int(np.log10(x))):g} \\times 10^{{{int(np.log10(x))}}}$"
            else:
                return f"${x:g}$"

        cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_fmt))

        plt.savefig(f'test_{idx+1}.png', dpi = 300)