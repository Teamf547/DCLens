# MMpS.py -- Magnification Map Simulation and Showing

# 2026-3-17 by Teamf
# the table of BHs' profiles delete the Einstein radius, and we import the mass instead
# the pc_to_redshift function is deleted, and we directly use pc as the distance unit

import os
import sys
import json
from .logger_config import setup_logger
logger = setup_logger("DCLens")
import numpy as np
import pandas as pd
import math

from .constants import pc_in_m
from .lensing_setup import split_into_planes
# NOTE(2026-05) 全 GPU 端到端：光栅化已集成到 lensing_core 的
# run_adaptive_rasterize 中，不再使用 CPU 端 moderngl 光栅化器。
# from .gpu_rasterizer import rasterize_triangles_to_texture_tiled

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
# import cv2

import time

from . import lensing_core

def BufferSizeProbe(ini_con_file,
                       D_L_pc=1e4, D_S_pc=1e9,
                       N_planes=1,
                       img_center=(0.0, 0.0), 
                       src_considered_region=(0.1, 0.1),
                       buffer_step=0.01, 
                       max_buffer_steps=200):
    """
    【独立 Buffer Size 评估模块】
    仅加载透镜平面并在 GPU 上执行 8-控制点多帧向外探针计算，
    快速返回推荐的保守 Buffer Size (arcsec)，不触发任何渲染。

    Parameters:
    -----------
    ini_con_file : str
        黑洞/透镜初始条件文件路径
    D_L_pc, D_S_pc : float
        透镜组和源平面的物理距离 (pc)
    N_planes : int
        多平面划分层数
    img_center : tuple (x, y)
        图像平面中心坐标 (arcsec)
    src_considered_region : tuple (width, height)
        目标源平面关注区域尺寸 (arcsec)
    buffer_step : float
        外扩探测步长 (arcsec)
    max_buffer_steps : int
        最大搜索帧数

    Returns:
    --------
    recommended_buffer : float
        推导出的保守 Buffer Size 推荐值 (arcsec)
    """
    logger.info("=== 启动 Buffer Size 独立预评估模块 ===")
    start_t = time.time()

    # 1. 加载并划分透镜平面数据
    df = pd.read_csv(
        ini_con_file, sep=r"\s+", header=None, skiprows=1,
        usecols=range(7),
        names=["Mass_[Msun]", "x_[pc]", "y_[pc]", "z_[pc]", "vx_[km/s]", "vy_[km/s]", "vz_[km/s]"]
    )
    df["z_[pc]"] = df["z_[pc]"] + D_L_pc
    df = df.loc[df["z_[pc]"] > 0].reset_index(drop=True)

    BH_table = pd.DataFrame({
        "x_pc": df["x_[pc]"],
        "y_pc": df["y_[pc]"],
        "M_Msun": df["Mass_[Msun]"],
        "z_pc": df["z_[pc]"],
    })
    
    lens_planes = split_into_planes(BH_table, N_planes=N_planes)
    logger.info(f"成功载入 {BH_table.shape[0]} 个黑洞，分布在 {len(lens_planes)} 个平面上。")

    # 2. 初始化 GPU 引擎并上传透镜数据
    engine = lensing_core.LensingEngineGPU(D_S_pc, 1000, 1000) # 仅用于探针，分配很小的队列容量即可
    
    lx_list, ly_list, lM_list = [], [], []
    z_list, factor_list = [], []
    for plane in lens_planes:
        z_list.append(plane['z_pc'])
        D_iS = D_S_pc - plane['z_pc']
        factor_list.append(D_iS / D_S_pc)
        lx_list.append(plane['lenses']['x_pc'])
        ly_list.append(plane['lenses']['y_pc'])
        lM_list.append(plane['lenses']['M_Msun'])
    
    engine.load_planes(lx_list, ly_list, lM_list, z_list, factor_list)

    # 3. 计算源平面目标区域边界
    xc, yc = img_center
    w, h = src_considered_region
    consider_xmin, consider_xmax = xc - w/2, xc + w/2
    consider_ymin, consider_ymax = yc - h/2, yc + h/2

    # 4. 调用 GPU 探针 Kernel
    recommended_buffer = engine.estimate_buffer_size(
        img_cx=xc, img_cy=yc, 
        base_w=w, base_h=h,
        src_xmin=consider_xmin, src_xmax=consider_xmax,
        src_ymin=consider_ymin, src_ymax=consider_ymax,
        buffer_step=buffer_step, 
        max_steps=max_buffer_steps
    )

    cost_t = time.time() - start_t
    logger.info(f"评估完成！耗时: {cost_t:.3f} 秒")
    logger.info(f"推荐的 Buffer Size: {recommended_buffer:.4f} arcsec")

    if recommended_buffer >= (buffer_step * max_buffer_steps):
        logger.warning("提示: 探针已达到最大搜索步数，建议增大 max_buffer_steps 或增加 buffer_step。")

    return recommended_buffer

# Magnification Map Simulation
def MagMapSim(ini_con_file,
            D_L_pc=1e4, D_S_pc=1e9,
            N_planes=1, coarse_ps=1e-3, finest_ps=1e-4,
            img_center=(0.0, 0.0), src_considered_region=(0.1, 0.1), buffer_size=0.1,
            max_refine_iters=20,
            srcp_ps=1e-4,
            # NOTE(2026-05) 全 GPU 端到端后，max_tile_dim/batch_triangles
            # 这些 moderngl 光栅化专属参数已不再使用，予以移除。
            # 但 ss 超采样因子仍在 GPU 端到端光栅化中支持。
            ss=1,
            output_mmp="mmp.npy", save_dir='.',
            max_coarse_per_chunk=200
            ):
    
    start_time = time.time()
    ## Step1: load and process lens plane data
    logger.info("loading and processing lens plane data...")
    # df = pd.read_csv(
    #     ini_con_file, sep=r"\s+", header=None, comment='#',
    #     skiprows=1,
    #     names=["Mass_[Msun]", "x_[pc]", "y_[pc]", "z_[pc]", "vx_[km/s]", "vy_[km/s]", "vz_[km/s]"]
    # )
    df = pd.read_csv(
    ini_con_file,
    sep=r"\s+", 
    header=None,
    skiprows=1,  # 跳过第一行
    usecols=range(7),  # 只读取前7列 (0-6)
    names=["Mass_[Msun]", "x_[pc]", "y_[pc]", "z_[pc]", "vx_[km/s]", "vy_[km/s]", "vz_[km/s]"]
    )
    df["z_[pc]"] = df["z_[pc]"] + D_L_pc
    df = df.loc[df["z_[pc]"] > 0].reset_index(drop=True)

    # NOTE(2026-05) 恢复原版薄透镜公式：ray_shoot_device 内部将透镜坐标
    # 经 ARCSEC2RAD 转换为 rad。因此这里必须把透镜物理坐标(pc)换算为角
    # 坐标(arcsec)后传给 GPU，才能与原版得到完全一致的放大率量级。
    BH_table = pd.DataFrame({
        "x_pc": df["x_[pc]"],
        "y_pc": df["y_[pc]"],
        "M_Msun": df["Mass_[Msun]"], # calculate einstein radius
        "z_pc": df["z_[pc]"],
    })
    
    lens_planes = split_into_planes(BH_table, N_planes=N_planes)
    # total_lenses = sum(len(p['lenses']['x']) for p in lens_planes)
    total_lenses = BH_table.shape[0]
    logger.info(f"total lenses number: {total_lenses}, distribute on {len(lens_planes)} planes.")
    

    ## step2: initialize GPU engine
    engine = lensing_core.LensingEngineGPU(D_S_pc, 20000000, 20000000) #giving large enough capacity for queue avoiding refine's break
    logger.info("transfer lenses data to GPU...")
    
    # prepare data struct
    lx_list, ly_list, lM_list = [], [], []
    z_list, factor_list = [], []

    for plane in lens_planes:
        z_list.append(plane['z_pc'])
        D_iS = D_S_pc - plane['z_pc']
        factor_list.append(D_iS / D_S_pc) # the projection factor
        lx_list.append(plane['lenses']['x_pc'])
        ly_list.append(plane['lenses']['y_pc'])
        lM_list.append(plane['lenses']['M_Msun'])
    
    # upload to GPU
    engine.load_planes(lx_list, ly_list, lM_list, z_list, factor_list)


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

    ## --- 2026-4-29 step4 & 5 & 6: 流式分块执行 GPU 四叉树细分与光栅化 (Streaming Pipeline)
    logger.info("running the GPU adaptive quadtree refinement with streaming spatial chunking...")
    
    cpp_start = time.time()
    
    # --- 【补充初始化】: 计算 ny_chunks 和 nx_chunks ---
    dx_chunk = max_coarse_per_chunk * coarse_ps
    dy_chunk = max_coarse_per_chunk * coarse_ps
    
    nx_chunks = int(np.ceil((img_xmax - img_xmin) / dx_chunk))
    ny_chunks = int(np.ceil((img_ymax - img_ymin) / dy_chunk))
    
    logger.info(f"Image plane split into {nx_chunks} x {ny_chunks} chunks to prevent GPU/CPU OOM.")

    # --- 提前计算源平面网格参数（为光栅化做准备） ---
    nx_src = int(np.ceil(w / srcp_ps))
    ny_src = int(np.ceil(h / srcp_ps))
    src_xmax = consider_xmin + nx_src * srcp_ps
    src_ymax = consider_ymin + ny_src * srcp_ps
    
    # 建立一张全局的、空的源平面累加图
    global_pixel_mu_map = np.zeros((ny_src, nx_src), dtype=np.float32)

    # 开始流式循环
    for iy in range(ny_chunks):
        for ix in range(nx_chunks):
            # 1. 计算当前 Chunk 的边界
            c_img_xmin = img_xmin + ix * dx_chunk
            c_img_xmax = min(img_xmin + (ix + 1) * dx_chunk, img_xmax)
            c_img_ymin = img_ymin + iy * dy_chunk
            c_img_ymax = min(img_ymin + (iy + 1) * dy_chunk, img_ymax)
            
            logger.info(f"Processing chunk ({ix+1}/{nx_chunks}, {iy+1}/{ny_chunks})...")

            # 2. 【全 GPU 端到端】在 GPU 上一次性完成：
            #    光线追踪 quadtree refine + 三角化光栅化 + 归一化。
            #    只返回最终的一整张源平面放大图 (ny_src, nx_src)，
            #    完全消除了中间叶子节点数据的 CPU<->GPU 往复 IO。
            #
            #    传入给 kernel 的源平面边界使用 (consider_xmin .. src_xmax) 之
            #    类区域，与原来 CPU 端光照化时的像素坐标归一化一致。
            chunk_map = engine.run_adaptive_gl_rasterize(
                c_img_xmin, c_img_xmax, c_img_ymin, c_img_ymax,
                coarse_ps, finest_ps,
                max_refine_iters,
                consider_xmin, src_xmax, consider_ymin, src_ymax,
                nx_src, ny_src,
                ss
            )

            # 3. 物理叠加：将该 Chunk 贡献的光度加到全局图上
            global_pixel_mu_map += chunk_map

            # 4. 释放本轮 chunk 的临时结果
            del chunk_map
            import gc
            gc.collect()

    cpp_end = time.time()
    logger.info(f"全部分块处理完毕 (GPU 端到端)。")

    # 将全局图赋值给最终结果变量
    pixel_mu_map = global_pixel_mu_map
    
    # 7. 保存结果
    # mag_normli = np.mean(pixel_mu_map)
    # pixel_mu_map = pixel_mu_map / mag_normli if mag_normli > 0 else pixel_mu_map
    
    os.makedirs(save_dir, exist_ok=True)
    magmap_outpath = os.path.join(save_dir, output_mmp)
    np.save(magmap_outpath, pixel_mu_map)
    
    # 构造元数据 (简化版)
    end_time = time.time()
    meta = {
        'run_info': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_duration_sec': round(end_time - start_time, 2),
            'cpp_refinement_duration_sec': round(cpp_end - cpp_start, 2),
            # 'total_leaf_nodes': len(mags_all),
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

    # 保存逻辑
    with open(magmap_outpath + ".meta.json", 'w') as fh:
        json.dump(meta, fh, indent=4) # indent=4 让 JSON 更易读
    
    logger.info(f"详细运行参数已记录至 {magmap_outpath}.meta.json")
    return {'pixel_mu_map': pixel_mu_map}

def MagMapShow(mmp_file, 
               ps, cut=0,
               img_center=(0.0, 0.0), 
               gamma=0.5,
               vmin=None, vmax=None,
               sys_name='N-body Lens System'):
    fig, axs = plt.subplots(1, 1, figsize = (8,6))

    H = np.load(mmp_file)
    H1 = H[cut:H.shape[0]-cut, cut:H.shape[1]-cut]
    x_len, y_len = H1.shape
    x1_min = img_center[0] - x_len * ps / 2
    x1_max = img_center[0] + x_len * ps / 2
    y1_min = img_center[1] - y_len * ps / 2
    y1_max = img_center[1] + y_len * ps / 2

    # log_mag = np.log10(H1+1e-6)
    # smoothed_log = cv2.bilateralFilter(log_mag.astype(np.float32), d=9, sigmaColor=0.3, sigmaSpace=75)
    # H1 = 10**smoothed_log

    if vmin is None:
        vmin1 = H1.min()
    else:
        vmin1 = vmin
    if vmax is None:
        vmax1 = H1.max()
    else:        
        vmax1 = vmax

    # norm1 = PowerNorm(gamma=gamma, vmin=vmin1, vmax=vmax1)
    norm1 = LogNorm(vmin=0.5, vmax=vmax1)

    im1 = axs.imshow(H1, origin='lower',
                    cmap='magma',
                    extent=[x1_min, x1_max, y1_max, y1_min],
                    aspect='auto', norm=norm1)

    axs.set_title(f'Magnification Map of SRC plane({sys_name})')
    axs.set_xlabel("βₓ (arcsec)")
    axs.set_ylabel("βᵧ (arcsec)")
    axs.set_aspect('equal', adjustable='box')

    cbar1 = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.046, pad=0.02)
    cbar1.set_label('Magnification')

    plt.show()