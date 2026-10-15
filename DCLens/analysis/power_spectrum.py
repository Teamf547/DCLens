"""Power spectra of curve ensembles, retaining the existing 365-day convention.

Input 'times' for these ensemble files is in years. LightCurveResult instead
stores time_days; convert explicitly before passing curves to these routines.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import signal
from tqdm import trange

def iter_valid_combinations(cfg):
    """根据 constraints 生成所有有效 (m, N, rh) 组合"""
    for m in cfg["m_list"]:
        for rh in cfg["rh_list"]:
            if (m, rh) not in cfg["constraints"]: #限制约束扫描参数空间
                continue
            min_N = cfg["constraints"][(m, rh)]
            for N in cfg["N_list"]:
                if N >= min_N:
                    yield m, N, rh

def combo_name(m, N, rh):
    return f"m{m}_N{N}_rh{rh}"

def resolve_latest_npz(out_dir, combo_name_str, r_px=5):
    pattern = os.path.join(out_dir, f"LC_{combo_name_str}_R{r_px}.npz")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]

def compute_group_psd_matrix(
    fluxes,
    times,
    psd_mode="fft",
    nperseg_default=256,
    n_freq_bins=120,
    welch_window="hann",
    welch_detrend="linear",
):
    """
    对一组光变曲线计算 PSD，并插值到统一对数频率网格。

    参数
    ----
    fluxes : (n_samples, T)
    times  : (T,)
        原单位为 years，内部转成 days
    psd_mode : "fft" 或 "welch"

    返回
    ----
    dict:
        freq_grid, X_log, median_log10, dt_days, fs, n_samples
    """
    psd_mode = psd_mode.lower().strip()
    if psd_mode not in {"fft", "welch"}:
        raise ValueError('psd_mode must be either "fft" or "welch".')

    n_samples, T = fluxes.shape
    if T < 2:
        raise ValueError("times 长度太短，无法计算 PSD。")

    dt_years = float(times[1] - times[0])
    dt_days = dt_years * 365
    if dt_days <= 0:
        raise ValueError(f"无效的时间步长 dt_days={dt_days}")

    fs = 1.0 / dt_days  # 频率单位：1/days

    psd_list = []
    global_fmin = np.inf
    global_fmax = 0.0

    for i in trange(n_samples, desc=f"Computing PSD ({psd_mode})", unit="lc", leave=False):
        y = np.asarray(fluxes[i], dtype=float)

        if psd_mode == "welch":
            nperseg = min(nperseg_default, T)
            noverlap = nperseg // 2
            f, Pxx = signal.welch(
                y,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window=welch_window,
                detrend=welch_detrend,
                scaling="density",
            )

        else:
            # FFT / periodogram
            x = y - np.mean(y)
            fft_vals = np.fft.rfft(x)
            f = np.fft.rfftfreq(T, d=dt_days)

            # one-sided PSD density
            Pxx = (dt_days / T) * (np.abs(fft_vals) ** 2)

            if T % 2 == 0:
                if len(Pxx) > 2:
                    Pxx[1:-1] *= 2.0
            else:
                if len(Pxx) > 1:
                    Pxx[1:] *= 2.0

        # 去掉 DC 分量
        if f[0] == 0.0:
            f_pos = f[1:]
            Pxx_pos = Pxx[1:]
        else:
            f_pos = f
            Pxx_pos = Pxx

        Pxx_pos = np.maximum(Pxx_pos, 1e-30)
        psd_list.append((f_pos, Pxx_pos))

        if f_pos.size > 0:
            global_fmin = min(global_fmin, float(f_pos.min()))
            global_fmax = max(global_fmax, float(f_pos.max()))

    if not np.isfinite(global_fmin) or global_fmin <= 0:
        global_fmin = 1e-12
    if global_fmin >= global_fmax:
        raise RuntimeError("无法构造有效的公共频率网格。")

    freq_grid = np.exp(np.linspace(np.log(global_fmin), np.log(global_fmax), n_freq_bins))
    log_freq_grid = np.log(freq_grid)

    X_log = np.zeros((n_samples, n_freq_bins), dtype=float)
    for i, (f_pos, Pxx_pos) in enumerate(psd_list):
        logf = np.log(np.maximum(f_pos, 1e-30))
        logP = np.log10(np.maximum(Pxx_pos, 1e-30))
        X_log[i, :] = np.interp(log_freq_grid, logf, logP)

    median_log10 = np.median(X_log, axis=0)

    return {
        "freq_grid": freq_grid,
        "X_log": X_log,
        "median_log10": median_log10,
        "dt_days": dt_days,
        "fs": fs,
        "n_samples": n_samples,
    }

def plot_psd_by_params(cfg, single_samples=False):
    """
    主函数：读取所有参数组合的 npz，计算 PSD，并画 median。
    返回 all_group_info，便于后续继续分析。
    """
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # 颜色 / 线型 / 点型：N / r_h / m
    N_values = cfg["N_list"]
    rh_values = cfg["rh_list"]
    m_values = cfg["m_list"]

    color_pool = list(plt.cm.tab10.colors)
    linestyle_pool = ["-", "--", "-.", ":"]
    marker_pool = ["o", "s", "^", "D", "v", "P", "X", "*"]

    color_map = {N: color_pool[i % len(color_pool)] for i, N in enumerate(N_values)}

    linestyle_map = {rh: linestyle_pool[i % len(linestyle_pool)] for i, rh in enumerate(rh_values)}
    marker_map = {m: marker_pool[i % len(marker_pool)] for i, m in enumerate(m_values)}

    all_group_info = {}

    print("Scanning valid parameter combinations and loading npz files...")

    for m, N, rh in iter_valid_combinations(cfg):
        cname = combo_name(m, N, rh)
        npz_path = resolve_latest_npz(cfg["out_dir"], cname, cfg["source_size_pixels"])

        if npz_path is None:
            print(f"[skip] npz not found for {cname}")
            continue

        print(f"[load] {cname} -> {npz_path}")

        with np.load(npz_path, allow_pickle=False) as data:
            if "fluxes" not in data.files or "times" not in data.files:
                raise ValueError(f"{npz_path} must contain 'fluxes' and 'times' arrays")

            fluxes = data["fluxes"]
            times = data["times"]

        group_psd = compute_group_psd_matrix(
            fluxes=fluxes,
            times=times,
            psd_mode=cfg["psd_mode"],
            nperseg_default=cfg["nperseg_default"],
            n_freq_bins=cfg["n_freq_bins"],
            welch_window=cfg.get("welch_window", "hann"),
            welch_detrend=cfg.get("welch_detrend", "linear"),
        )

        all_group_info[cname] = {
            "m": m,
            "N": N,
            "rh": rh,
            "npz_path": npz_path,
            **group_psd,
        }

        # 每组 median 保存 CSV
        df = pd.DataFrame({
            "freq_1_per_day": group_psd["freq_grid"],
            "median_log10_psd": group_psd["median_log10"],
        })
        csv_path = os.path.join(cfg["out_dir"], f"psd_median_{cname}_{cfg['psd_mode']}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  saved median csv -> {csv_path}")

    if not all_group_info:
        raise RuntimeError("没有找到任何可用的参数组合对应的 npz 文件。")

    # 画图
    fig, ax = plt.subplots(figsize=cfg["figsize"])

    for cname, info in all_group_info.items():
        m = info["m"]
        N = info["N"]
        rh = info["rh"]

        color = color_map[N]
        linestyle = linestyle_map[rh]
        marker = marker_map[m]

        freq_grid = info["freq_grid"]
        median_log10 = info["median_log10"]

        if cfg["xscale"] == "log":
            ax.semilogx(
                freq_grid,
                median_log10,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=cfg["markevery"],
                markersize=cfg["markersize"],
                linewidth=cfg["line_width"],
                alpha=cfg["line_alpha"],
            )
        else:
            ax.plot(
                freq_grid,
                median_log10,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=cfg["markevery"],
                markersize=cfg["markersize"],
                linewidth=cfg["line_width"],
                alpha=cfg["line_alpha"],
            )

        if single_samples:
            n_samples = np.random.randint(0, info['n_samples'], size=cfg.get('n_single_samples',3))
            for idx in n_samples:
                if idx < info["n_samples"]:
                    single_log10 = info["X_log"][idx]
                    if cfg["xscale"] == "log":
                        ax.semilogx(
                            freq_grid,
                            single_log10,
                            color=color,
                            markevery=cfg["markevery"],
                            markersize=cfg["markersize"] * 0.8,
                            linewidth=1.0,
                            alpha=0.3,
                            label="single sample PSD" if idx == n_samples[0] else None,
                        )
                    else:
                        ax.plot(
                            freq_grid,
                            single_log10,
                            color=color,
                            markevery=cfg["markevery"],
                            markersize=cfg["markersize"] * 0.8,
                            linewidth=1.0,
                            alpha=0.3,
                            label="single sample PSD" if idx == n_samples[0] else None,
                        )

    ax.set_xlabel("Frequency (1/days)", fontsize=cfg["label_fontsize"])
    ax.set_ylabel(f"$\\log_{{10}}(PSD)$", fontsize=cfg["label_fontsize"])
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # 顶部留空间给三行 legend
    fig.subplots_adjust(top=0.72)

    handles_N = [
        Line2D([0], [0], color=color_map[N], lw=2.0, label=f"N={N}")
        for N in N_values
    ]
    handles_rh = [
        Line2D([0], [0], color="black", linestyle=linestyle_map[rh], lw=2.0, label=rf"$r_h={rh}$")
        for rh in rh_values
    ]
    handles_m = [
        Line2D([0], [0], color="black", marker=marker_map[m], linestyle="None",
               markersize=7, label=f"m={m}")
        for m in m_values
    ]

    fig.legend(
        handles=handles_N,
        title="N",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(handles_N),
        frameon=False,
        fontsize=cfg["legend_fontsize"],
        title_fontsize=cfg["legend_title_fontsize"],
    )
    fig.legend(
        handles=handles_rh,
        title=r"$r_h$",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(handles_rh),
        frameon=False,
        fontsize=cfg["legend_fontsize"],
        title_fontsize=cfg["legend_title_fontsize"],
    )
    fig.legend(
        handles=handles_m,
        title="m",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=len(handles_m),
        frameon=False,
        fontsize=cfg["legend_fontsize"],
        title_fontsize=cfg["legend_title_fontsize"],
    )

    out_png = os.path.join(cfg["out_dir"], cfg["output_fig_name"])
    plt.savefig(out_png, dpi=cfg["dpi"], bbox_inches="tight", pad_inches=0.25)
    plt.show()

    print("Saved figure to:", out_png)

    # 总表
    summary_rows = []
    for cname, info in all_group_info.items():
        summary_rows.append({
            "combo": cname,
            "m": info["m"],
            "N": info["N"],
            "rh": info["rh"],
            "npz_path": info["npz_path"],
            "n_samples": info["n_samples"],
            "dt_days": info["dt_days"],
            "fs_1_per_day": info["fs"],
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(cfg["out_dir"], f"psd_summary_{cfg['psd_mode']}.csv"), index=False)
    print("Saved summary table.")

    return all_group_info

def plot_single_npz_psd_with_1sigma(npz_path, cfg):
    """
    读取单一 npz 文件，计算 PSD，并绘制：
    1. 中位数 (Median PSD)
    2. 1-sigma 置信区间 (使用 16% 和 84% 分位数)
    3. 随机选取的 3 条单次采样 PSD (Legend 中仅标一次)
    """
    print(f"[load] Loading {npz_path}...")
    with np.load(npz_path, allow_pickle=False) as data:
        if "fluxes" not in data.files or "times" not in data.files:
            raise ValueError(f"{npz_path} 必须包含 'fluxes' 和 'times' 数组")
        
        fluxes = data["fluxes"]
        times = data["times"]

    # 1. 计算 PSD 矩阵
    group_psd = compute_group_psd_matrix(
        fluxes=fluxes,
        times=times,
        psd_mode=cfg.get("psd_mode", "fft"),
        nperseg_default=cfg.get("nperseg_default", 256),
        n_freq_bins=cfg.get("n_freq_bins", 120),
        welch_window=cfg.get("welch_window", "hann"),
        welch_detrend=cfg.get("welch_detrend", "linear"),
    )

    freq_grid = group_psd["freq_grid"]
    X_log = group_psd["X_log"]           # shape: (n_samples, n_freq_bins)
    median_log10 = group_psd["median_log10"]
    n_samples = group_psd["n_samples"]

    # 2. 计算 1-sigma 置信区间 (基于对数空间的 16th 和 84th 分位数)
    p16 = np.percentile(X_log, 16, axis=0)
    p84 = np.percentile(X_log, 84, axis=0)

    # 3. 随机抽取 3 条成员曲线
    num_to_select = min(3, n_samples)
    random_indices = np.random.choice(n_samples, size=num_to_select, replace=False)

    # 4. 开始绘图
    fig, ax = plt.subplots(figsize=cfg.get("figsize", (8, 6)))

    # 绘制 1-sigma 阴影区
    ax.fill_between(
        freq_grid,
        p16,
        p84,
        color="#f1605d",
        alpha=0.3,
        label=r"$1\sigma$ Range"
    )

    # 绘制 3 条随机的单次采样 PSD
    for i, idx in enumerate(random_indices):
        single_log10 = X_log[idx]
        # 只为第一条曲线添加 label，保证 legend 中只出现一次
        label = "Single sample PSD" if i == 0 else None
        
        if cfg.get("xscale", "log") == "log":
            ax.semilogx(freq_grid, single_log10, color="#721f81", 
                        alpha=0.6, linewidth=1.0, label=label)
        else:
            ax.plot(freq_grid, single_log10, color="#721f81", 
                    alpha=0.6, linewidth=1.0, label=label)

    # 绘制 Median PSD
    if cfg.get("xscale", "log") == "log":
        ax.semilogx(freq_grid, median_log10, color="#150e37", 
                    linewidth=2.0, label="Median PSD")
    else:
        ax.plot(freq_grid, median_log10, color="#150e37", 
                linewidth=2.0, label="Median PSD")

    # 5. 图形设置
    ax.set_xlabel("Frequency (1/days)", fontsize=cfg.get("label_fontsize", 12))
    ax.set_ylabel(r"$\log_{10}(\mathrm{PSD})$", fontsize=cfg.get("label_fontsize", 12))
    
    title = cfg.get("title", f"PSD Analysis for {os.path.basename(npz_path)}")
    ax.set_title(title, fontsize=cfg.get("title_fontsize", 14))
    
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    
    # 图例设置
    ax.legend(fontsize=cfg.get("legend_fontsize", 10), loc="best", frameon=False)

    plt.tight_layout()

    # 如果有配置保存路径，则保存图片
    if "out_dir" in cfg and "output_fig_name" in cfg:
        os.makedirs(cfg["out_dir"], exist_ok=True)
        out_png = os.path.join(cfg["out_dir"], cfg["output_fig_name"])
        plt.savefig(out_png, dpi=cfg.get("dpi", 300), bbox_inches="tight")
        print(f"Saved figure to: {out_png}")

    plt.show()

    return group_psd
