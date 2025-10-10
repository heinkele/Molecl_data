"""
Process nanopore events using ONLY peak_fits/fit_results.json

What it does:
- Finds each sample's peak_fits/fit_results.json
- For every peak, loads trace.t, trace.raw, trace.filtered, trace.fitted (if present)
- Detects event window on filtered
- Computes baselines (RAW), AUC, and delta metrics:
    Δ(filtered, raw) and Δ(fitted, raw) with a linear mapping to RAW units
- Saves per-sample CSV: event_features_with_deltas.csv
- Saves plots under sample_dir/plots/
"""

# ===================== CONFIG =====================
ROOT = "/Users/hugo/MOLECL_test/Molecl_data_H"

# Detector (signature → two edge peaks)
ROLL_WIN         = 31
PEAK_HEIGHT_MULT = 8.0
REL_HEIGHT       = 0.5
GUARD_SAMPLES    = 2
MIN_SPAN         = 8
CENTER_TIME_HINT = 0.0
USE_PEAK_TOPS    = True

# Baseline & AUC
BASELINE_MODE    = "median"   # "median" | "average" | "linear"
AUC_MODE         = "flat"     # "flat" uses pooled baseline; "linear" uses pre→post line
DEFICIT_ONLY     = False      # True for strict ECD (deficit only)

# Plots (separate folders)
SAVE_PLOT_ISO        = True   # event isolation (no AUC shading)
SAVE_PLOT_AUC        = True   # AUC shading figure
SAVE_PLOT_DELTA_FILT = True   # delta(filtered, raw)
SAVE_PLOT_DELTA_FIT  = True   # delta(fit, raw) if fit available

PLOT_EVERY            = 50     # plot 1 out of N events
MAX_PLOTS_PER_FOLDER  = 12
ZOOM_MODE             = False
ZOOM_PAD_SEC          = 0.001


# ===================== IMPORTS =====================
import os, json, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.stats import skew, kurtosis


# ===================== HELPERS =====================
def get_data_folders(root_path):
    folders = []
    for item in os.listdir(root_path):
        if item.startswith('.') or item == '__pycache__':
            continue
        full_path = os.path.join(root_path, item)
        if os.path.isdir(full_path) and 'baseline' not in item.lower():
            folders.append(full_path)
    return sorted(folders)


def find_fit_jsons(root):
    fit_jsons = []
    for sample_dir in get_data_folders(root):
        fit_path = os.path.join(sample_dir, "peak_fits", "fit_results.json")
        if os.path.exists(fit_path):
            fit_jsons.append(fit_path)
    return sorted(fit_jsons)


def ensure_odd(n):
    n = int(n)
    return n + 1 if (n % 2 == 0) else n


def movavg(x, win):
    if win <= 1:
        return x
    win = ensure_odd(win)
    pad = win // 2
    xpad = np.pad(x, pad, mode="reflect")
    ker = np.ones(win) / float(win)
    return np.convolve(xpad, ker, mode="valid")


def roll_std_centered(x, win):
    if win <= 1:
        return np.zeros_like(x)
    win = ensure_odd(win)
    m1 = movavg(x, win)
    m2 = movavg(x * x, win)
    var = np.maximum(m2 - m1 * m1, 0.0)
    return np.sqrt(var)


def change_signature(filtered, win):
    f = filtered - np.median(filtered)
    rmean = movavg(f, win)
    rstd = roll_std_centered(f, win)
    rstd_mean = roll_std_centered(rmean, win)
    sig = rstd * rstd_mean
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    smax = np.max(sig) if sig.size else 0.0
    return (sig / smax) if smax > 0 else sig


def center_index_from_time(time, center_time=CENTER_TIME_HINT):
    if len(time) == 0:
        return 0
    if time[0] <= center_time <= time[-1]:
        return int(np.argmin(np.abs(time - center_time)))
    return len(time) // 2


def movavg3(x):
    x = np.asarray(x, float)
    if x.size < 3:
        return x.copy()
    k = np.array([1.0, 1.0, 1.0]) / 3.0
    xpad = np.pad(x, 1, mode="edge")
    return np.convolve(xpad, k, mode="valid")


# ---------- Detector (two edge peaks around center) ----------
def detect_bounds_signature_centered(filtered, time,
                                     win=ROLL_WIN,
                                     peak_height_mult=PEAK_HEIGHT_MULT,
                                     rel_height=REL_HEIGHT,
                                     guard=GUARD_SAMPLES,
                                     min_span=MIN_SPAN,
                                     center_time=CENTER_TIME_HINT,
                                     use_peak_tops=USE_PEAK_TOPS,
                                     peak_guard=2):
    sig = change_signature(filtered, win=win)
    cidx = center_index_from_time(time, center_time=center_time)

    height = peak_height_mult * np.median(sig) if sig.size else 0.0
    distance = max(5, win // 2)
    peaks, _ = find_peaks(sig, height=height, distance=distance, prominence=0.05)

    if peaks.size == 0:
        left = max(0, cidx - max(min_span // 2, 1))
        right = min(len(filtered) - 1, cidx + max(min_span // 2, 1))
        return left, right, cidx, sig, cidx, cidx

    left_candidates = peaks[peaks <= cidx]
    right_candidates = peaks[peaks >= cidx]

    if left_candidates.size == 0 or right_candidates.size == 0:
        pk = int(peaks[np.argmin(np.abs(peaks - cidx))])
        widths, _, L, R = peak_widths(sig, [pk], rel_height=rel_height)
        left = max(0, int(np.floor(L[0])) - guard)
        right = min(len(filtered) - 1, int(np.ceil(R[0])) + guard)
        if (right - left) < min_span:
            half = max(min_span // 2, 1)
            left = max(0, pk - half)
            right = min(len(filtered) - 1, pk + half)
        return left, right, pk, sig, pk, pk

    left_peak  = int(left_candidates[np.argmax(left_candidates)])
    right_peak = int(right_candidates[np.argmin(right_candidates)])

    if use_peak_tops:
        left  = max(0, left_peak  - max(peak_guard, guard))
        right = min(len(filtered) - 1, right_peak + max(peak_guard, guard))
    else:
        wL, _, L_ips, _ = peak_widths(sig, [left_peak],  rel_height=rel_height)
        wR, _, _, R_ips = peak_widths(sig, [right_peak], rel_height=rel_height)
        left  = max(0, int(np.floor(L_ips[0])) - guard)
        right = min(len(filtered) - 1, int(np.ceil(R_ips[0])) + guard)

    if (right - left) < min_span:
        mid = (left + right) // 2
        half = max(min_span // 2, 1)
        left  = max(0, mid - half)
        right = min(len(filtered) - 1, mid + half)

    central_ref = int((left_peak + right_peak) // 2)
    return left, right, central_ref, sig, left_peak, right_peak


# ---------- Baseline from window RAW edges ----------
def real_baseline_from_raw(raw, start, end,
                           guard_pre=200, guard_post=200,
                           trim_q=0.10, min_samples=20,
                           mode=BASELINE_MODE):
    n = len(raw)
    i0, i1 = max(0, start - guard_pre), start
    j0, j1 = end + 1, min(n, end + 1 + guard_post)

    pre = raw[i0:i1].astype(float, copy=False)
    post = raw[j0:j1].astype(float, copy=False)

    if pre.size < min_samples:
        i0 = max(0, start - (guard_pre + (min_samples - pre.size)))
        pre = raw[i0:i1]
    if post.size < min_samples:
        j1 = min(n, end + 1 + (guard_post + (min_samples - post.size)))
        post = raw[j0:j1]

    def _trimmed_median(x, q=trim_q):
        if x.size == 0:
            return np.nan
        lo, hi = np.quantile(x, [q, 1.0 - q])
        sel = x[(x >= lo) & (x <= hi)]
        return float(np.median(sel if sel.size else x))

    b_pre  = _trimmed_median(pre)
    b_post = _trimmed_median(post)
    drift  = float(b_post - b_pre) if np.isfinite(b_pre) and np.isfinite(b_post) else np.nan

    if mode == "average":
        b_evt = float(np.nanmean([b_pre, b_post]))
    elif mode == "linear" and np.isfinite(b_pre) and np.isfinite(b_post):
        x = np.r_[np.arange(i0, i1), np.arange(j0, j1)]
        y = np.r_[pre, post]
        if x.size >= 2:
            p = np.polyfit(x, y, 1)
            mid = 0.5 * (start + end)
            b_evt = float(p[0] * mid + p[1])
        else:
            b_evt = float(np.nanmean([b_pre, b_post]))
    else:
        pooled = np.r_[pre, post]
        b_evt = _trimmed_median(pooled)

    return b_pre, b_post, b_evt, drift


# ---------- AUC on window RAW ----------
def compute_auc_abs(raw, start, end, dt, baseline, deficit_only=DEFICIT_ONLY):
    seg = raw[start:end+1].astype(float)
    if np.isscalar(baseline):
        dev = seg - float(baseline)
    else:
        base_arr = np.asarray(baseline, float)
        if base_arr.shape[0] != seg.shape[0]:
            raise ValueError("baseline array must match event length")
        dev = seg - base_arr
    if deficit_only:
        return float(np.trapz(np.maximum(-dev, 0.0), dx=dt))
    return float(np.trapz(np.abs(dev), dx=dt))


# ---------- Linear mapping & deltas ----------
def linear_map_and_predict(x, y, start, end):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    seg = slice(start, end+1)
    xs, ys = x[seg], y[seg]
    if xs.size == 0 or np.allclose(np.std(xs), 0.0, atol=1e-12):
        a, b = 0.0, float(np.median(ys)) if ys.size else 0.0
        yhat = np.full_like(y, b, dtype=float)
        return a, b, yhat
    A = np.vstack([xs, np.ones_like(xs)]).T
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(a), float(b), (a * x + b)


def delta_metrics(y_true, y_hat, dt, start, end):
    seg = slice(start, end+1)
    diff = y_true[seg].astype(float) - y_hat[seg].astype(float)
    mae = float(np.mean(np.abs(diff))) if diff.size else float("nan")
    area = float(np.trapz(np.abs(diff), dx=dt)) if diff.size else float("nan")
    return area, mae

# ---------- Skewness and kurtosis ----------

def skew_kurtosis_fisher(x):
    """
    Fisher–Pearson sample skewness and excess kurtosis (SciPy implementation).
    Returns (skewness, excess_kurtosis).
    NaN if input too short or not finite.
    """
    x = np.asarray(x, float)
    if x.size < 3 or not np.any(np.isfinite(x)):
        return float("nan"), float("nan")

    # Skewness: Fisher–Pearson, unbiased
    g1 = skew(x, bias=False, nan_policy="omit")

    # Kurtosis: Fisher=True → excess kurtosis (normal = 0.0)
    g2 = kurtosis(x, fisher=True, bias=False, nan_policy="omit")

    return float(g1), float(g2)



# ===================== PLOTTING =====================
def save_plot_isolation(out_png, time, raw, filtered, start, end, signature, zoom=True):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios":[2,2,1]})
    # filtered
    axes[0].plot(time, filtered, label="Filtered", color="C0")
    axes[0].axvspan(time[start], time[end], color="orange", alpha=0.25, label="Event")
    axes[0].set_ylabel("Filtered (a.u.)"); axes[0].legend(loc="upper right")
    # raw
    axes[1].plot(time, raw, label="Raw", color="C2")
    axes[1].axvspan(time[start], time[end], color="orange", alpha=0.25)
    axes[1].set_ylabel("Raw (units)"); axes[1].legend(loc="upper right")
    # signature
    axes[2].plot(time, signature, color="C4", label="Change signature")
    axes[2].axvspan(time[start], time[end], color="orange", alpha=0.25)
    axes[2].set_ylabel("Sig (norm.)"); axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")
    if zoom:
        tmin = float(time[start]) - ZOOM_PAD_SEC
        tmax = float(time[end])   + ZOOM_PAD_SEC
        for ax in axes: ax.set_xlim(tmin, tmax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close(fig)


def save_plot_auc(out_png, time, raw, filtered, start, end, signature,
                  b_pre, b_post, b_evt, auc_abs, auc_mode=AUC_MODE, zoom=True):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios":[2,2,1]})
    axes[0].plot(time, filtered, label="Filtered", color="C0")
    axes[0].axvspan(time[start], time[end], color="orange", alpha=0.25, label="Event")
    axes[0].set_ylabel("Filtered (a.u.)"); axes[0].legend(loc="upper right")
    # raw + shaded area
    axes[1].plot(time, raw, label="Raw", color="C2")
    if auc_mode == "linear":
        base = np.linspace(b_pre, b_post, end - start + 1)
        axes[1].plot(time[start:end+1], base, color="k", lw=1.2, ls="--", label="Baseline (linear)")
        y1 = raw[start:end+1]; y2 = base
    else:
        axes[1].axhline(b_evt, color="k", lw=1.2, ls="--", label="Baseline (flat)")
        y1 = raw[start:end+1]; y2 = np.full_like(y1, b_evt, dtype=float)
    axes[1].fill_between(time[start:end+1], y1, y2, alpha=0.25, color="C3",
                         label=f"AUC abs = {auc_abs:.3g}")
    axes[1].axvspan(time[start], time[end], color="orange", alpha=0.20)
    axes[1].set_ylabel("Raw (units)"); axes[1].legend(loc="upper right")
    axes[2].plot(time, signature, color="C4", label="Change signature")
    axes[2].axvspan(time[start], time[end], color="orange", alpha=0.25)
    axes[2].set_ylabel("Sig (norm.)"); axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")
    if zoom:
        tmin = float(time[start]) - ZOOM_PAD_SEC
        tmax = float(time[end])   + ZOOM_PAD_SEC
        for ax in axes: ax.set_xlim(tmin, tmax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close(fig)


def plot_delta(out_png, time, raw, curve, label_curve,
               start, end, zoom=True, note=""):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(time, raw,  label="Raw",   color="C2", lw=1.2)
    ax.plot(time, curve, label=label_curve, color="C0", lw=1.2)
    ax.fill_between(time[start:end+1], raw[start:end+1], curve[start:end+1],
                    color="C3", alpha=0.25, label="|Δ| area")
    ax.axvspan(time[start], time[end], color="orange", alpha=0.2, zorder=0)
    if note: ax.set_title(note)
    ax.set_ylabel("Current (raw units)"); ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    if zoom:
        tmin = float(time[start]) - ZOOM_PAD_SEC
        tmax = float(time[end])   + ZOOM_PAD_SEC
        ax.set_xlim(tmin, tmax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140); plt.close(fig)


# ===================== CORE =====================
def load_fits(fit_json_path):
    """Load peak_fits/fit_results.json into a dict keyed by peak_index."""
    with open(fit_json_path, "r") as fh:
        data = json.load(fh)
    rows = data if isinstance(data, list) else (list(data.values()) if isinstance(data, dict) else [])
    out = {}
    for d in rows:
        k = d.get("peak_index", d.get("signal_type_index"))
        if k is None:
            continue
        try:
            out[int(k)] = d
        except Exception:
            continue
    return out


def process_fit_json(fit_json_path):
    # fit_json_path: .../<sample>/peak_fits/fit_results.json
    sample_dir = os.path.dirname(os.path.dirname(fit_json_path))
    print("Processing (fits only):", fit_json_path)
    fit_map = load_fits(fit_json_path)
    if not fit_map:
        print("  (no fit rows)")
        return

    # outputs
    out_csv  = os.path.join(sample_dir, "event_features_with_deltas.csv")
    plot_root         = os.path.join(sample_dir, "plots")
    dir_iso           = os.path.join(plot_root, "event_isolation")
    dir_auc           = os.path.join(plot_root, "auc")
    dir_delta_filt    = os.path.join(plot_root, "delta_filtered")
    dir_delta_fit     = os.path.join(plot_root, "delta_fit")
    for d in [dir_iso, dir_auc, dir_delta_filt, dir_delta_fit]:
        os.makedirs(d, exist_ok=True)

    # CSV header
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "fit_json","event_idx",
            "start_idx","end_idx",
            "t_start","t_end","duration_s",
            "peak_idx","roll_win","rel_height","height_mult",
            "raw_baseline_pre","raw_baseline_post","raw_baseline_event","raw_drift_pp",
            "auc_abs",
            "average_blockage","maximum_blockage",
            "delta_area_filtered_raw","delta_mae_filtered_raw",
            "delta_area_fit_raw","delta_mae_fit_raw",
            "map_a_filtered","map_b_filtered",
            "map_a_fit","map_b_fit",
            "fit_available",
            "skewness_raw_resid","kurtosis_excess_raw_resid"  
        ])

    # plot counters
    n_iso = n_auc = n_df = n_dfit = 0

    # iterate peaks in index order
    for i in sorted(fit_map.keys()):
        try:
            payload = fit_map[i]
            tr = payload.get("trace", {})

            # Pull arrays (NEW schema). If absent, we skip; you can add OLD/NPZ fallback if needed.
            t        = np.array(tr.get("t", []), float)
            raw      = np.array(tr.get("raw", []), float)
            filtered = np.array(tr.get("filtered", []), float)
            fitted   = np.array(tr.get("fitted", []), float)

            if not (t.size and raw.size and filtered.size):
                print(f"  peak {i}: missing trace arrays; skipped")
                continue

            # Ensure equal length (guard)
            n = min(t.size, raw.size, filtered.size, fitted.size if fitted.size else t.size)
            t, raw, filtered = t[:n], raw[:n], filtered[:n]
            if fitted.size:
                fitted = fitted[:n]

            # dt
            dt = float(np.median(np.diff(t))) if n >= 2 else 1.0

            # detect bounds on filtered
            start, end, peak_idx, sig, _, _ = detect_bounds_signature_centered(
                filtered, t,
                win=ROLL_WIN,
                peak_height_mult=PEAK_HEIGHT_MULT,
                rel_height=REL_HEIGHT,
                guard=GUARD_SAMPLES,
                min_span=MIN_SPAN,
                center_time=CENTER_TIME_HINT,
                use_peak_tops=USE_PEAK_TOPS,
                peak_guard=GUARD_SAMPLES
            )

            # baselines (RAW)
            b_pre, b_post, b_evt, drift_pp = real_baseline_from_raw(
                raw, start, end,
                guard_pre=200, guard_post=200,
                trim_q=0.10, min_samples=20,
                mode=BASELINE_MODE
            )

            # AUC (RAW)
            if AUC_MODE == "linear":
                baseline_for_auc = np.linspace(b_pre, b_post, end - start + 1)
            else:
                baseline_for_auc = b_evt
            auc_abs = compute_auc_abs(raw, start, end, dt, baseline_for_auc, deficit_only=DEFICIT_ONLY)

            # average_blockage & maximum_blockage
            seg_raw = raw[start:end+1].astype(float)
            if AUC_MODE == "linear":
                base_arr = np.linspace(b_pre, b_post, end - start + 1)
                deficit = base_arr - seg_raw
            else:
                deficit = b_evt - seg_raw
            duration = float((end - start) * dt)
            average_blockage = float(auc_abs / duration) if duration > 0 else float("nan")
            peak_deficit_smoothed = float(np.max(movavg3(np.abs(deficit)))) if deficit.size else float("nan")
            maximum_blockage = float(auc_abs / peak_deficit_smoothed) if peak_deficit_smoothed > 0 else float("nan")

            # Map filtered -> raw (robust to offset/scale differences)
            a_f, b_f, filtered_phys = linear_map_and_predict(filtered, raw, start, end)
            d_area_fr, d_mae_fr = delta_metrics(raw, filtered_phys, dt, start, end)

            # Map fitted -> raw (if available)
            fit_available = False
            a_fit = b_fit = np.nan
            d_area_fit = d_mae_fit = np.nan
            fit_curve_phys = None

            if fitted.size:
                a_fit, b_fit, fit_curve_phys = linear_map_and_predict(fitted, raw, start, end)
                d_area_fit, d_mae_fit = delta_metrics(raw, fit_curve_phys, dt, start, end)
                fit_available = True

            if AUC_MODE == "linear":
                base_for_mom = np.linspace(b_pre, b_post, end - start + 1)
            else:
                base_for_mom = np.full_like(seg_raw, b_evt, dtype=float)
            resid = seg_raw - base_for_mom  # baseline-corrected RAW segment
            skew_raw, kurt_ex_raw = skew_kurtosis_fisher(resid)

            # write CSV row
            t_start = float(t[start]); t_end = float(t[end])
            with open(out_csv, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    os.path.basename(fit_json_path), i,
                    start, end, t_start, t_end, duration,
                    peak_idx, ROLL_WIN, REL_HEIGHT, PEAK_HEIGHT_MULT,
                    b_pre, b_post, b_evt, drift_pp,
                    auc_abs,
                    average_blockage, maximum_blockage,
                    d_area_fr, d_mae_fr,
                    d_area_fit, d_mae_fit,
                    a_f, b_f,
                    a_fit, b_fit,
                    int(fit_available),
                    skew_raw, kurt_ex_raw
                ])

            # plots (same logic as before)
            make_plot = (i % PLOT_EVERY == 0)
            if make_plot and SAVE_PLOT_ISO and n_iso < MAX_PLOTS_PER_FOLDER:
                out_iso = os.path.join(dir_iso, f"event_{i:05d}.png")
                save_plot_isolation(out_iso, t, raw, filtered, start, end, sig, zoom=ZOOM_MODE)
                n_iso += 1
            if make_plot and SAVE_PLOT_AUC and n_auc < MAX_PLOTS_PER_FOLDER:
                out_auc = os.path.join(dir_auc, f"event_{i:05d}.png")
                save_plot_auc(out_auc, t, raw, filtered, start, end, sig,
                              b_pre, b_post, b_evt, auc_abs, auc_mode=AUC_MODE, zoom=ZOOM_MODE)
                n_auc += 1
            if make_plot and SAVE_PLOT_DELTA_FILT and n_df < MAX_PLOTS_PER_FOLDER:
                out_df = os.path.join(dir_delta_filt, f"event_{i:05d}.png")
                note = f"Δ(filtered, raw): area={d_area_fr:.3g} pA·s, MAE={d_mae_fr:.3g} pA"
                plot_delta(out_df, t, raw, filtered_phys, "Mapped filtered",
                           start, end, zoom=ZOOM_MODE, note=note)
                n_df += 1
            if make_plot and SAVE_PLOT_DELTA_FIT and fit_available and n_dfit < MAX_PLOTS_PER_FOLDER:
                out_dfit = os.path.join(dir_delta_fit, f"event_{i:05d}.png")
                note2 = f"Δ(fit, raw): area={d_area_fit:.3g} pA·s, MAE={d_mae_fit:.3g} pA"
                plot_delta(out_dfit, t, raw, fit_curve_phys, "Mapped fit",
                           start, end, zoom=ZOOM_MODE, note=note2)
                n_dfit += 1

        except Exception as e:
            print(f"  peak {i}: ERROR -> {e}")

    print("  -> CSV:", out_csv)
    print("  -> Plots:", plot_root)


# ===================== DRIVER =====================
def __main__():
    fit_jsons = find_fit_jsons(ROOT)
    if not fit_jsons:
        print("No peak_fits/fit_results.json found under:", ROOT)
        return

    print(f"Found {len(fit_jsons)} fit json file(s).")
    for fpath in fit_jsons:
        process_fit_json(fpath)
    print("Done.")

if __name__ == '__main__':
    __main__()
