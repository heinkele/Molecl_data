# =========================================================
# ======================= CONFIG ==========================
# =========================================================
CONFIG = {
    # ---- IO / Structure ----
    "ROOT_PATH": "/Users/hugo/New data/PacBio",
    "INPUT_JSON_NAME": "combined_peaks_data.json",
    "RESULTS_SUBDIR": "peak_fits",
    "HIST_SUBDIR": "histograms",
    "SKIP_IF_FIT_EXISTS": True,         # skip a folder if fit_results.json already exists

    # ---- Plotting ----
    "SAVE_PLOTS": True,
    "FIGSIZE": (10, 6),
    "FIG_DPI": 150,
    "PLOT_RAW": True,
    "PLOT_FILTERED": True,
    "PLOT_FITTED": True,
    "XLABEL_TIME": "Time (µs)",

    # ---- Histograms / Stats ----
    "FWHM_HIST_BINS": 50,
    "AREA_HIST_BINS": 300,
    "AREA_XLIM": (0, 20.5),             # set to None for auto limits
    "SAVE_STATS_TXT": True,
    "SAVE_CORR": True,

    # ---- Optional post-fit filtering (for stats/plots only) ----
    "MIN_FWHM": None,                   # e.g., 2.0 (µs)
    "MAX_FWHM": None,                   # e.g., 30.0
    "MIN_AREA": None,                   # e.g., 0.02
    "MAX_AREA": None,
    "MIN_MAX_DISPLACEMENT": None,       # e.g., 0.01
    "MAX_MAX_DISPLACEMENT": None,

    # ---- Refined fit model & robustness ----
    "FIT_MODEL": "gaussian",            # "gaussian" | "supergauss" | "skewgauss"
    "SUPER_GAUSS_P": 2,               # only used if FIT_MODEL == "supergauss" (2 == Gaussian)
    "ROBUST_LOSS": "linear",           # "linear" | "soft_l1" | "huber" | "cauchy"
    "ROBUST_F_SCALE": 1.0,

    # ---- Parameter bounds (units: µs) ----
    "AMP_MIN": 0.05,                
    "AMP_MAX": 0.9,                 
    "SIGMA_MIN": 0.0,               
    "SIGMA_MAX": 10.0,              
    "MU_PAD_US": 20.0,                

    # ---- Optional window refinement around the max before fitting ----
    "REFIT_USE_CENTERED_WINDOW": True,
    "REFIT_HALF_WINDOW_US": 50,       
}

# =========================================================
# ==================== IMPLEMENTATION =====================
# =========================================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from scipy.optimize import least_squares

# Project-local imports
from screening_sample_ssd_old import get_data_folders
from process_single_peak import process_single_peak

# --------------------- Utilities -------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def to_native(x):
    """Convert numpy types to plain Python for JSON serialization."""
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def apply_optional_filters(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Filter dataframe by optional config cutoffs (for stats/plots only)."""
    filtered = df.copy()
    if cfg["MIN_FWHM"] is not None:
        filtered = filtered[filtered["fwhm"] >= cfg["MIN_FWHM"]]
    if cfg["MAX_FWHM"] is not None:
        filtered = filtered[filtered["fwhm"] <= cfg["MAX_FWHM"]]
    if cfg["MIN_AREA"] is not None:
        filtered = filtered[filtered["area"] >= cfg["MIN_AREA"]]
    if cfg["MAX_AREA"] is not None:
        filtered = filtered[filtered["area"] <= cfg["MAX_AREA"]]
    if cfg["MIN_MAX_DISPLACEMENT"] is not None:
        filtered = filtered[filtered["max_displacement"] >= cfg["MIN_MAX_DISPLACEMENT"]]
    if cfg["MAX_MAX_DISPLACEMENT"] is not None:
        filtered = filtered[filtered["max_displacement"] <= cfg["MAX_MAX_DISPLACEMENT"]]
    return filtered

# ------------------- Fit models --------------------------
def model_gaussian(t, A, mu, sigma, B):
    return A * np.exp(-0.5 * ((t - mu) / max(sigma, 1e-12))**2) + B

def model_supergauss(t, A, mu, sigma, B, p):
    # p=2 -> Gaussian; p=4 -> flatter top
    return A * np.exp(-0.5 * (np.abs((t - mu) / max(sigma, 1e-12))**p)) + B

def model_skewgauss(t, A, mu, sigma, B, alpha):
    # Simple skewed Gaussian via normal CDF factor
    z = (t - mu) / max(sigma, 1e-12)
    cdf = 0.5 * (1.0 + np.erf(alpha * z / np.sqrt(2)))
    return A * np.exp(-0.5 * z**2) * np.clip(2*cdf, 1e-6, 2.0) + B

def initial_guess(t, y):
    """
    Robust initial guesses for A, mu, sigma, B.
    A0 ≈ (max - median); mu0 at argmax; sigma0 from half-height width (or window/6).
    """
    y_med = float(np.median(y))
    y_pk  = float(np.max(y))
    A0 = max(1e-9, y_pk - y_med)
    mu0 = float(t[np.argmax(y)])

    # Light smoothing for half-height width
    if len(y) >= 9:
        y_s = np.convolve(y, np.ones(7)/7, mode='same')
    else:
        y_s = y
    half = y_med + 0.5 * (y_pk - y_med)
    above = np.where(y_s >= half)[0]
    if len(above) >= 2:
        fwhm = float(t[above[-1]] - t[above[0]])
        sigma0 = max(1e-3, fwhm / 2.355)
    else:
        sigma0 = max(1e-3, (t[-1] - t[0]) / 6.0)

    B0 = y_med
    return A0, mu0, sigma0, B0

def refine_fit_peak(t, y, cfg: Dict):
    """
    Robust baseline-including peak fit on (t, y), where t is in µs and y is the
    chosen trace (e.g., filtered * norm_factor).
    Returns dict with A, mu, sigma, B, (alpha), fwhm, r2, area, t, yfit.
    """
    # Optional: re-center to a local window around the maximum
    if cfg.get("REFIT_USE_CENTERED_WINDOW", False):
        mu_guess = t[np.argmax(y)]
        half = cfg.get("REFIT_HALF_WINDOW_US", (t[-1] - t[0]) / 2.0)
        msk = (t >= mu_guess - half) & (t <= mu_guess + half)
        if msk.sum() >= 8:
            t = t[msk]
            y = y[msk]

    A0, mu0, s0, B0 = initial_guess(t, y)

    mdl = str(cfg.get("FIT_MODEL", "gaussian")).lower()
    if mdl == "supergauss":
        p = float(cfg.get("SUPER_GAUSS_P", 4.0))
        def residuals(pvec):
            return model_supergauss(t, pvec[0], pvec[1], pvec[2], pvec[3], p) - y
        p0 = np.array([A0, mu0, s0, B0], dtype=float)
        lb = np.array([cfg["AMP_MIN"], t.min()-cfg["MU_PAD_US"], cfg["SIGMA_MIN"], B0 - 5*abs(A0+1e-9)])
        ub = np.array([cfg["AMP_MAX"], t.max()+cfg["MU_PAD_US"], cfg["SIGMA_MAX"], B0 + 5*abs(A0+1e-9)])

    elif mdl == "skewgauss":
        def residuals(pvec):
            return model_skewgauss(t, pvec[0], pvec[1], pvec[2], pvec[3], pvec[4]) - y
        p0 = np.array([A0, mu0, s0, B0, 1.5], dtype=float)   # alpha initial
        lb = np.array([cfg["AMP_MIN"], t.min()-cfg["MU_PAD_US"], cfg["SIGMA_MIN"], B0 - 5*abs(A0+1e-9), -5.0])
        ub = np.array([cfg["AMP_MAX"], t.max()+cfg["MU_PAD_US"], cfg["SIGMA_MAX"], B0 + 5*abs(A0+1e-9),  5.0])

    else:  # gaussian
        def residuals(pvec):
            return model_gaussian(t, pvec[0], pvec[1], pvec[2], pvec[3]) - y
        p0 = np.array([A0, mu0, s0, B0], dtype=float)
        lb = np.array([cfg["AMP_MIN"], t.min()-cfg["MU_PAD_US"], cfg["SIGMA_MIN"], B0 - 5*abs(A0+1e-9)])
        ub = np.array([cfg["AMP_MAX"], t.max()+cfg["MU_PAD_US"], cfg["SIGMA_MAX"], B0 + 5*abs(A0+1e-9)])

    # Ensure p0 starts inside bounds
    eps = 1e-12
    p0 = np.minimum(np.maximum(p0, lb + eps), ub - eps)

    res = least_squares(
        residuals, p0, bounds=(lb, ub),
        loss=cfg.get("ROBUST_LOSS", "soft_l1"),
        f_scale=cfg.get("ROBUST_F_SCALE", 1.0),
        max_nfev=20000
    )
    p = res.x

    if mdl == "skewgauss":
        A, mu, sigma, B, alpha = [float(v) for v in p]
        yfit = model_skewgauss(t, A, mu, sigma, B, alpha)
    elif mdl == "supergauss":
        A, mu, sigma, B = [float(v) for v in p]
        yfit = model_supergauss(t, A, mu, sigma, B, float(cfg.get("SUPER_GAUSS_P", 4.0)))
        alpha = None
    else:
        A, mu, sigma, B = [float(v) for v in p]
        yfit = model_gaussian(t, A, mu, sigma, B)
        alpha = None

    rss = float(np.sum((y - yfit)**2))
    tss = float(np.sum((y - np.mean(y))**2)) if len(y) > 1 else np.nan
    r2  = float(1 - rss / tss) if tss > 0 else np.nan

    fwhm = float(2.355 * sigma)  # Gaussian relation

    if mdl == "gaussian":
        area = float(A * sigma * np.sqrt(2*np.pi))
    else:
        area = float(np.trapz(yfit - B, t))

    out = {
        "A": A, "mu": mu, "sigma": sigma, "B": B,
        "fwhm": fwhm, "r2": r2, "area": area,
        "t": t, "yfit": yfit
    }
    if alpha is not None:
        out["alpha"] = alpha
    return out

# ------------------ Main processing ----------------------
def process_combined_peaks(root_path: str, cfg: Dict):
    data_folders = get_data_folders(root_path)
    print("Data folders:", data_folders)

    for base_dir in data_folders:
        input_json = os.path.join(base_dir, cfg["INPUT_JSON_NAME"])
        results_dir = os.path.join(base_dir, cfg["RESULTS_SUBDIR"])
        hist_dir = os.path.join(results_dir, cfg["HIST_SUBDIR"])
        fit_json_out = os.path.join(results_dir, "fit_results.json")

        if cfg["SKIP_IF_FIT_EXISTS"] and os.path.exists(fit_json_out):
            print(f"Skipping (already fitted): {base_dir}")
            continue

        if not os.path.exists(input_json):
            print(f"  Missing JSON in {base_dir} → {cfg['INPUT_JSON_NAME']} not found.")
            continue

        ensure_dir(results_dir)
        ensure_dir(hist_dir)

        # ---- Load combined peaks JSON ----
        with open(input_json, "r") as f:
            peaks_data = json.load(f)
        combined_df = pd.DataFrame(peaks_data)

        print(f"\nProcessing: {base_dir}")
        print("Columns:", list(combined_df.columns))

        # ---- Build peak windows via your per-peak pre-processing ----
        fitted_results = process_single_peak(combined_df)

        fit_results = []
        for idx, item in enumerate(fitted_results):
            try:
                # === Pull data from process_single_peak output ===
                # Full window time axis (already in µs, centered ~0 around the event)
                t_signal = np.asarray(item["t_signal"], dtype=float)  # shape (N,)

                # This is the filtered / normalized feature-like signal used for fitting,
                # after centering (what you're plotting in blue)
                filtered_signal = np.asarray(item["signal_array"], dtype=float)  # shape (N,)

                # Raw current trace for that same full window
                raw_signal_not_norm = np.asarray(item["raw_signal_not_norm"], dtype=float)  # shape (N,)

                norm_factor = float(item["norm_factor"])
                signal_type = item["signal_type"]

                # We'll fit on the same full-window arrays for now
                t_fit = t_signal  # µs
                y_fit = filtered_signal.astype(float)  # a.u.; same length as t_signal

                # Optional debug
                if idx == 0:
                    med_dt = float(np.median(np.diff(t_fit))) if len(t_fit) > 1 else float('nan')
                    print(
                        f"[debug] t_fit range: {t_fit.min():.2f} → {t_fit.max():.2f} µs; "
                        f"median dt ≈ {med_dt:.3f} µs"
                    )

                # === Run the robust fitter (gives us a *cropped* fit and params)
                fit = refine_fit_peak(t_fit, y_fit, cfg)
                # fit is expected to have:
                #   fit["t"]      -> time (µs) in the refined-fit subwindow
                #   fit["yfit"]   -> model prediction on that subwindow
                #   fit["A"], fit["B"], fit["mu"], fit["sigma"], fit["fwhm"], fit["area"], etc.

                # ---- Pull fit metrics ----
                t_event          = float(fit["mu"])        # µs
                integral         = float(fit["area"])
                max_displacement = float(fit["A"])
                fwhm             = float(fit["fwhm"])      # µs
                r2               = float(fit["r2"])
                baseline_B       = float(fit["B"])
                amp_A            = float(fit["A"])
                mu_center        = float(fit["mu"])        # µs
                sigma_width      = float(fit["sigma"])     # µs
                alpha_skew       = float(fit["alpha"]) if "alpha" in fit else None

                # === RESAMPLE THE FIT BACK ONTO THE FULL WINDOW ===
                # fit["t"] and fit["yfit"] live only on a tight subwindow around the event peak.
                # We need to put that model back on the full t_signal axis so downstream
                # code can compare raw vs fitted sample-by-sample.

                t_fit_local   = np.asarray(fit["t"], dtype=float)      # µs (short window)
                yfit_local    = np.asarray(fit["yfit"], dtype=float)   # same length as t_fit_local

                # We'll build fitted_resampled with same length as t_signal:
                if t_fit_local.size >= 2:
                    # interpolate model onto full t_signal grid
                    fitted_resampled = np.interp(
                        t_signal,           # query points (full window timebase, µs)
                        t_fit_local,        # known x (cropped fit window, µs)
                        yfit_local,         # known y
                        left=np.nan,
                        right=np.nan
                    )
                else:
                    # degenerate fallback
                    fitted_resampled = np.full_like(t_signal, np.nan, dtype=float)

                # === BUILD THE ROW WE'LL SAVE ===
                # SUPER IMPORTANT:
                # Everything in "trace" must be SAME LENGTH and SAME TIMEBASE.
                # We'll use t_signal (µs) as that canonical time axis.

                fit_row = {
                    "source_file": base_dir,
                    "peak_index": idx,

                    # main fitted parameters
                    "t_event_us": t_event,               # event center in µs
                    "area": integral,
                    "max_displacement": max_displacement,
                    "fwhm_us": fwhm,                     # FWHM in µs
                    "r2": r2,
                    "baseline": baseline_B,
                    "amplitude": amp_A,
                    "center_us": mu_center,              # redundant with t_event_us, but nice to have
                    "sigma_us": sigma_width,
                    "signal_type": signal_type,
                    "model": cfg.get("FIT_MODEL", "gaussian"),
                }

                if alpha_skew is not None:
                    fit_row["alpha"] = alpha_skew

                # TRACE BLOCK (canonicalized)
                fit_row["trace"] = {
                    # full window time axis, in µs
                    "t":        t_signal.tolist(),

                    # raw current in that same window
                    "raw":      raw_signal_not_norm.astype(float).tolist(),

                    # filtered/feature signal you actually fit (blue curve). Do NOT rescale or swap here.
                    "filtered": y_fit.astype(float).tolist(),

                    # fitted model projected back to that same window
                    "fitted":   fitted_resampled.astype(float).tolist(),
                }

                # Append to list
                fit_results.append(fit_row)


                # ---------- Per-peak plot (two panels: RAW on top, FEATURE+FIT bottom) ----------
                if cfg["SAVE_PLOTS"]:
                    # --- Build a robust raw trace for plotting ---
                    raw = np.asarray(item.get("raw_signal_not_norm", []), float)
                    # If raw is missing/flat, try to reconstruct: raw ≈ raw_signal * median(norm_signal)
                    if raw.size != t_signal.size or np.nanstd(raw) < 1e-12 or np.allclose(raw, 0.0):
                        raw_alt  = np.asarray(item.get("raw_signal", []), float)      # often I / I_norm (dimensionless)
                        norm_sig = np.asarray(item.get("norm_signal", []), float)      # baseline (Amps)
                        if raw_alt.size and norm_sig.size:
                            scale = float(np.nanmedian(norm_sig))
                            raw = raw_alt * scale
                        else:
                            # last resort: scale feature so you see something (purely for visualization)
                            raw = y_fit * 0.0

                    # Make sure lengths match (interpolate raw to t_signal if needed)
                    if raw.size != t_signal.size and raw.size > 1:
                        # assume both cover same time range; linear interp
                        t_raw = np.linspace(t_signal.min(), t_signal.max(), raw.size)
                        raw = np.interp(t_signal, t_raw, raw)

                    # --- Figure layout: RAW on top, FEATURE+FIT on bottom ---
                    fig = plt.figure(figsize=cfg["FIGSIZE"], dpi=cfg["FIG_DPI"])
                    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
                    ax_raw = fig.add_subplot(gs[0])
                    ax_feat = fig.add_subplot(gs[1], sharex=ax_raw)

                    # Top: RAW
                    ax_raw.plot(t_signal, raw, "k-", linewidth=1.0, label="Raw")
                    ax_raw.set_ylabel("Raw (A)")
                    ax_raw.legend(loc="upper right")
                    # Hide x tick labels on the top panel
                    plt.setp(ax_raw.get_xticklabels(), visible=False)

                    # Bottom: FEATURE + FIT
                    if cfg["PLOT_FILTERED"]:
                        ax_feat.plot(t_fit, y_fit, "b-", label="Filtered (fit input)")
                    if cfg["PLOT_FITTED"]:
                        ax_feat.plot(fit["t"], fit["yfit"], "r--",
                                    label=f'Fit ({cfg.get("FIT_MODEL","gaussian")}, R²={r2:.3f})')
                    ax_feat.set_xlabel(cfg.get("XLABEL_TIME", "Time"))
                    ax_feat.set_ylabel("Feature (a.u.)")
                    ax_feat.legend(loc="upper right")

                    # Common title
                    fig.suptitle(
                        f"Peak fit - {os.path.basename(base_dir)} - Peak {idx}\n"
                        f"FWHM={fwhm:.2f} µs, A={max_displacement:.3f}, B={baseline_B:.3f}",
                        y=0.98
                    )

                    fig.tight_layout()
                    fig.subplots_adjust(top=0.88)
                    #plt.xlim(-0.1, 0.1)
                    plt.savefig(os.path.join(results_dir, f"fit_peak_{idx}.png"))
                    plt.close(fig)


            except Exception as e:
                print(f"  Error processing peak {idx} in {base_dir}: {e}")
                continue

        # ---- Save table (CSV + JSON) ----
        fit_df = pd.DataFrame(fit_results)
        fit_df.to_csv(os.path.join(results_dir, "fit_results.csv"), index=False)

        fit_results_native = [{k: to_native(v) for k, v in row.items()} for row in fit_results]
        with open(fit_json_out, "w") as f:
            json.dump(fit_results_native, f, indent=2)

        # ---- Stats / Plots (with optional filtering) ----
        if fit_df.empty:
            print("  No fits to summarize.")
            continue

        fit_df_f = apply_optional_filters(fit_df, cfg)
        if fit_df_f.empty:
            print("  All fits filtered out by thresholds; no stats computed.")
            continue

        # FWHM histogram
        if "fwhm" in fit_df_f.columns and cfg["SAVE_PLOTS"]:
            plt.figure(figsize=cfg["FIGSIZE"], dpi=cfg["FIG_DPI"])
            plt.hist(fit_df_f["fwhm"].dropna(), bins=cfg["FWHM_HIST_BINS"], edgecolor="black")
            plt.title("Histogram of Peak FWHM")
            plt.xlabel("FWHM (µs)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, "fwhm_histogram.png"))
            plt.close()

            if cfg["SAVE_STATS_TXT"]:
                with open(os.path.join(hist_dir, "fwhm_statistics.txt"), "w") as f:
                    f.write("FWHM Statistics:\n")
                    f.write(f"Mean FWHM: {fit_df_f['fwhm'].mean():.6f}\n")
                    f.write(f"Std  FWHM: {fit_df_f['fwhm'].std():.6f}\n")
                    f.write(f"Median FWHM: {fit_df_f['fwhm'].median():.6f}\n")

                print("\nFWHM Statistics:")
                print(f"Mean:   {fit_df_f['fwhm'].mean():.6f}")
                print(f"Std:    {fit_df_f['fwhm'].std():.6f}")
                print(f"Median: {fit_df_f['fwhm'].median():.6f}")

        # Area histogram + stats
        areas = fit_df_f["area"].to_numpy(float)
        if cfg["SAVE_PLOTS"] and areas.size > 0:
            plt.figure(figsize=cfg["FIGSIZE"], dpi=cfg["FIG_DPI"])
            plt.hist(areas, bins=cfg["AREA_HIST_BINS"], edgecolor="black")
            if cfg["AREA_XLIM"] is not None:
                plt.xlim(*cfg["AREA_XLIM"])
            plt.title("Histogram of Peak Areas")
            plt.xlabel("Area (a.u.)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, "area_histogram.png"))
            plt.close()

        if cfg["SAVE_STATS_TXT"]:
            with open(os.path.join(hist_dir, "area_statistics.txt"), "w") as f:
                f.write("Area Statistics:\n")
                f.write(f"Mean Area: {np.mean(areas):.6f}\n")
                f.write(f"Std Area:  {np.std(areas):.6f}\n")
                f.write(f"Median Area:{np.median(areas):.6f}\n")

        print("\nArea Statistics:")
        print(f"Mean Area:   {np.mean(areas):.6f}")
        print(f"Std Area:    {np.std(areas):.6f}")
        print(f"Median Area: {np.median(areas):.6f}")

        # Correlations
        if cfg["SAVE_CORR"] and {"fwhm", "max_displacement"}.issubset(fit_df_f.columns):
            if cfg["SAVE_PLOTS"]:
                # FWHM vs Max Displacement
                plt.figure(figsize=cfg["FIGSIZE"], dpi=cfg["FIG_DPI"])
                plt.scatter(fit_df_f["fwhm"], fit_df_f["max_displacement"], alpha=0.5)
                plt.title("FWHM vs Max Displacement")
                plt.xlabel("FWHM (µs)")
                plt.ylabel("Max Displacement")
                plt.tight_layout()
                plt.savefig(os.path.join(hist_dir, "fwhm_vs_max_displacement.png"))
                plt.close()

                # FWHM vs Area
                plt.figure(figsize=cfg["FIGSIZE"], dpi=cfg["FIG_DPI"])
                plt.scatter(fit_df_f["fwhm"], areas, alpha=0.5)
                plt.title("FWHM vs Area")
                plt.xlabel("FWHM (µs)")
                plt.ylabel("Area (a.u.)")
                plt.tight_layout()
                plt.savefig(os.path.join(hist_dir, "fwhm_vs_area.png"))
                plt.close()

                # Max Displacement vs Area
                plt.figure(figsize=cfg["FIGSIZE"], dpi=cfg["FIG_DPI"])
                plt.scatter(fit_df_f["max_displacement"], areas, alpha=0.5)
                plt.title("Max Displacement vs Area")
                plt.xlabel("Max Displacement")
                plt.ylabel("Area (a.u.)")
                plt.tight_layout()
                plt.savefig(os.path.join(hist_dir, "max_displacement_vs_area.png"))
                plt.close()

            with open(os.path.join(hist_dir, "correlation_statistics.txt"), "w") as f:
                f.write("Correlation Statistics:\n")
                f.write(f"FWHM vs Max Displacement: {np.corrcoef(fit_df_f['fwhm'], fit_df_f['max_displacement'])[0,1]:.6f}\n")
                f.write(f"FWHM vs Area:             {np.corrcoef(fit_df_f['fwhm'], areas)[0,1]:.6f}\n")
                f.write(f"Max Disp vs Area:         {np.corrcoef(fit_df_f['max_displacement'], areas)[0,1]:.6f}\n")

            print("\nCorrelation Statistics:")
            print(f"FWHM vs Max Displacement: {np.corrcoef(fit_df_f['fwhm'], fit_df_f['max_displacement'])[0,1]:.6f}")
            print(f"FWHM vs Area:             {np.corrcoef(fit_df_f['fwhm'], areas)[0,1]:.6f}")
            print(f"Max Displacement vs Area: {np.corrcoef(fit_df_f['max_displacement'], areas)[0,1]:.6f}")

        print("\nProcessing complete!")
        print(f"Results saved in: {results_dir}")

# ------------------------ Main ---------------------------
def __main__():
    process_combined_peaks(CONFIG["ROOT_PATH"], CONFIG)

if __name__ == "__main__":
    __main__()
