"""Observation-matched random paths and Lomb-Scargle comparisons.

This separate analysis retains its existing four-sigma full-map source model,
observational resampling and noise assumptions. It is not the interactive or
sequence light-curve backend. Optional dependencies: scipy and astropy.
"""
import numpy as np
import pandas as pd
import os
import logging
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
# 1. 核心数学与物理函数

def make_pure_gaussian_kernel(r_px):
    r = int(r_px)
    if r < 1:
         return np.array([[1.0]], dtype=float), 1
    sigma = r / 2.0
    grid_r = int(np.ceil(4 * sigma)) 
    y, x = np.ogrid[-grid_r:grid_r+1, -grid_r:grid_r+1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel, grid_r

def process_one_map(map_arr, kernel, r_px):
    conv = fftconvolve(map_arr, kernel, mode='same')
    H, W = conv.shape
    x_min, x_max = r_px, W - r_px
    y_min, y_max = r_px, H - r_px
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"裁剪区域太小，有效宽:{x_max-x_min}, 有效高:{y_max-y_min}")
    valid_x = np.arange(x_min, x_max)
    valid_y = np.arange(y_min, y_max)
    return conv, valid_x, valid_y

def is_signal_significant(flux, cfg):
    f_max, f_min, f_mean = np.max(flux), np.min(flux), np.mean(flux)
    if cfg["filter_mode"] == "absolute":
        return (f_max - f_min) >= cfg["signal_threshold"]
    else:
        denom = max(f_mean, 1e-12)
        return ((f_max - f_min) / denom) >= cfg["signal_threshold"]

def prepare_observation(csv_path):
    df = pd.read_csv(csv_path).sort_values('mjd')
    best_band = df['band'].value_counts().idxmax()
    obs = df[df['band'] == best_band].copy()
    
    t_mjd = obs['mjd'].values
    t_rel = t_mjd - t_mjd.min()
    
    mag = obs['mag'].values
    mag_err = obs['magerr'].values
    mag_base = np.median(mag)
    flux_obs = 10**(-0.4 * (mag - mag_base))
    flux_err = flux_obs * (0.4 * np.log(10) * mag_err)
    
    return t_rel, flux_obs, flux_err, best_band

def extract_obs_matched_curves(conv_map, valid_x, valid_y, t_rel, flux_err_obs, cfg):
    pm_per_day = cfg["proper_motion"] / 365.25
    total_days = t_rel.max()
    total_path_px = (total_days * pm_per_day) / cfg["pixel_scale"]
    actual_path_len = int(np.ceil(total_path_px)) + 5 
    
    H, W = conv_map.shape
    max_start_x = len(valid_x) - actual_path_len
    if max_start_x <= 0:
        raise ValueError(f"Map 宽度不足。需要 {actual_path_len} px，但 valid_x 只有 {len(valid_x)}")

    rng = np.random.default_rng(cfg["random_seed"])
    sim_flux_list = []
    chosen_ys = rng.choice(valid_y, size=cfg["n_paths"], replace=(len(valid_y) < cfg["n_paths"]))
    avg_noise = np.median(flux_err_obs)

    for y in chosen_ys:
        start_x_idx = rng.integers(0, max_start_x + 1)
        space_slice = conv_map[y, valid_x[start_x_idx : start_x_idx + actual_path_len]]
        x_space = np.arange(len(space_slice))
        t_space = x_space * (cfg["pixel_scale"] / pm_per_day)
        
        interp_f = interp1d(t_space, space_slice, kind='linear', bounds_error=False, fill_value="extrapolate")
        sampled_flux = interp_f(t_rel)
        sampled_flux /= np.median(sampled_flux)
        
        if is_signal_significant(sampled_flux, cfg):
            noise = rng.normal(0, avg_noise, size=len(sampled_flux))
            sim_flux_list.append(sampled_flux + noise)
            
    return np.array(sim_flux_list)
# 2. 物理判据与批量绘图逻辑

def calculate_criteria(f_obs, sim_fluxes, freqs, psd_obs, psd_sims):
    median_psd = np.median(psd_sims, axis=0)
    p16, p84 = np.percentile(psd_sims, [16, 84], axis=0)
    sigma_sim = np.std(psd_sims, axis=0)
    pull = (psd_obs - median_psd) / np.where(sigma_sim==0, 1e-10, sigma_sim)
    
    red_chi2 = np.sum(pull**2) / len(freqs)
    overlap = np.sum((psd_obs >= p16) & (psd_obs <= p84)) / len(freqs) * 100
    obs_std = np.std(f_obs)
    sim_mean_std = np.mean(np.std(sim_fluxes, axis=1))
    
    # 严格判据判定
    is_candidate = "YES" if (red_chi2 < 1.3 and overlap > 75 and (0.5 < obs_std/sim_mean_std < 2.0)) else "NO"
    
    return red_chi2, overlap, obs_std, sim_mean_std, pull, median_psd, p16, p84, is_candidate

def plot_and_save(t_obs, f_obs, f_err, sim_fluxes, freqs, psd_obs, median_psd, p16, p84, pull, map_name, save_dir):
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.errorbar(t_obs, f_obs, yerr=f_err, fmt='o', color='black', label='Observed Data', markersize=3, alpha=0.5)
    rng = np.random.default_rng()
    indices = rng.choice(len(sim_fluxes), size=min(3, len(sim_fluxes)), replace=False)
    for i, idx in enumerate(indices):
        ax1.plot(t_obs, sim_fluxes[idx], alpha=0.7, label=f'Sim Sample {i+1}', lw=1.2)
    ax1.set_title(f"Lightcurve Match")
    ax1.set_xlabel("Time (Days)")
    ax1.set_ylabel(f"$\\mathcal{{M}}$(Magnification of total flux)")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.loglog(freqs, median_psd, color='blue', label='Sim Median', lw=2)
    ax2.fill_between(freqs, p16, p84, color='blue', alpha=0.2, label='1-sigma Range')
    ax2.loglog(freqs, psd_obs, color='red', label='Observed PSD', alpha=0.8)
    ax2.set_title("PSD Match")
    ax2.set_xlabel("Frequency (1/Days)")
    ax2.set_ylabel(f"$log_{{10}}(Power)$")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(pull, bins=25, color='steelblue', edgecolor='white')
    ax3.axvline(0, color='red', linestyle='--')
    ax3.set_title(r"PSD Residuals ($\sigma$)")
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Number of Frequencies Bins")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{map_name}_match.png"), dpi=150)
    plt.close(fig)
# 3. 批量运行主逻辑

def OBSandSIM_Match(cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logging.info(f"Starting batch process for observation: {cfg['obs_csv']}")

    # 1. 【循环外预计算】
    t_rel, f_obs, f_err, band = prepare_observation(cfg["obs_csv"])
    
    f_min = 1.0 / (t_rel.max() + 1)
    f_max = 0.5 / np.median(np.diff(t_rel))
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), 150)
    psd_obs = LombScargle(t_rel, f_obs).power(freqs)
    
    kernel, k_radius = make_pure_gaussian_kernel(cfg["source_size_pixels"])
    results = []

    # 2. 遍历参数空间，并加入 constraints 约束逻辑
    for m in cfg["m_list"]:
        for rh in cfg["rh_list"]:
            key = (m, rh)
            
            # 如果当前 (m, rh) 组合不在约束字典中，说明该组合不被允许/不产生数据，直接跳过
            if key not in cfg["constraints"]:
                logging.debug(f"Skip: (m={m}, rh={rh}) not in constraints.")
                continue
                
            max_N = cfg["constraints"][key]
            
            for Ns in cfg["N_list"]:
                # 核心约束逻辑：当前 N 必须小于等于约束中设定的最大 N
                if Ns > max_N:
                    continue
                
                map_name = f"m_{m}_N{Ns}_rh_{rh}"
                map_path = os.path.join(cfg["base_dir"], f"{map_name}.npy")
                
                if not os.path.exists(map_path):
                    continue
                
                logging.info(f"Evaluating {map_name}...")
                
                # 读取与处理 Map
                cmap = np.load(map_path)
                if cfg["replace_small_values"]:
                    cmap[cmap <= cfg["small_value_threshold"]] = 1.0
                
                try:
                    conv_map, v_x, v_y = process_one_map(cmap, kernel, k_radius)
                    sim_fluxes = extract_obs_matched_curves(conv_map, v_x, v_y, t_rel, f_err, cfg)
                except ValueError as e:
                    logging.error(f"Failed to process {map_name}: {e}")
                    continue
                
                if len(sim_fluxes) == 0:
                    logging.warning(f"No significant signal in {map_name}")
                    results.append({"Map": map_name, "m": m, "N": Ns, "rh": rh, "Candidate": "NO_SIGNAL"})
                    continue

                # 计算判据
                psd_sims = np.array([LombScargle(t_rel, s).power(freqs) for s in sim_fluxes])
                red_chi2, overlap, obs_std, sim_mean_std, pull, med_psd, p16, p84, is_cand = calculate_criteria(
                    f_obs, sim_fluxes, freqs, psd_obs, psd_sims
                )
                
                results.append({
                    "Map": map_name,
                    "m": m, "N": Ns, "rh": rh,
                    "Chi_Square_v": round(red_chi2, 4),
                    "Overlap_Pct": round(overlap, 2),
                    "Obs_Std": round(obs_std, 4),
                    "Sim_Std": round(sim_mean_std, 4),
                    "Candidate": is_cand
                })
                
                if cfg["save_plots"]:
                    plot_and_save(t_rel, f_obs, f_err, sim_fluxes, freqs, psd_obs, 
                                  med_psd, p16, p84, pull, map_name, cfg["output_dir"])

    # 3. 汇总输出
    if results:
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(cfg["output_dir"], cfg["output_csv"])
        df_results.to_csv(csv_path, index=False)
        logging.info(f"Batch evaluation complete! Report saved to {csv_path}")
        
        candidates = df_results[df_results["Candidate"] == "YES"]
        if not candidates.empty:
            print("\n*** SUCCESS! Found Matching Parameters ***")
            print(candidates[["Map", "Chi_Square_v", "Overlap_Pct", "Sim_Std"]])
        else:
            print("\n--- No Golden Candidates Found in this parameter space ---")