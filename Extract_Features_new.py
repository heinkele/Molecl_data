# ===================== CONFIG =====================

ROOT = "/Users/hugo/New data/PacBio"

# Peak window detector parameters
ROLL_WIN         = 31        # rolling window for change_signature
PEAK_HEIGHT_MULT = 8.0       # multiplier on median(signature) to define "strong" peaks
REL_HEIGHT       = 0.5       # rel_height for scipy.signal.peak_widths
GUARD_SAMPLES    = 2         # pad a few samples left/right around detected edges
MIN_SPAN         = 8         # minimum number of samples in the event slice
CENTER_TIME_HINT = 0.0       # we assume t=0 is near the event
USE_PEAK_TOPS    = True      # True: cut window around peak tops instead of FWHM edges

# Baseline & AUC options
BASELINE_MODE    = "median"  # "median" | "average" | "linear"
AUC_MODE         = "flat"    # "flat"   | "linear"
DEFICIT_ONLY     = False     # if True, only integrate downward excursions

PRE_POST_SAMPLES      = 200   # samples for each side (before/after)
PRE_POST_MIN_SAMPLES  = 20    # minimum per-side to accept; otherwise that side ignored
MERGE_PRE_POST_TO_SINGLE = True  # keep True (we output only merged features)
EVAL_PRE_POST_PLOTS  = True   # whether to compute pre/post plots (can be slow)

# Plot controls
SAVE_PLOT_ISO        = True  # filtered/raw/sig with event box
SAVE_PLOT_AUC        = True  # AUC shading plot
SAVE_PLOT_DELTA_FILT = True  # delta(filtered→raw mapping) in-event
SAVE_PLOT_DELTA_FIT  = True  # delta(fit→raw mapping) in-event
PLOT_EVERY           = 50    # only plot 1 out of N events
MAX_PLOTS_PER_FOLDER = 12    # per plot type
ZOOM_MODE            = False # if True, zoom each plot around event
ZOOM_PAD_SEC         = 0.001 # zoom padding in seconds for the x-limits


# ===================== IMPORTS =====================

import os, json, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.stats import skew, kurtosis

# ===================== HELPERS =====================

def get_data_folders(root_path):
    """
    Return list of subfolders under ROOT that look like samples.
    """
    folders = []
    for item in os.listdir(root_path):
        if item.startswith('.') or item == '__pycache__':
            continue
        full_path = os.path.join(root_path, item)
        if os.path.isdir(full_path) and 'baseline' not in item.lower():
            folders.append(full_path)
    return sorted(folders)

def find_fit_jsons(root):
    """
    Return list of .../sampleX/peak_fits/fit_results.json that exist.
    """
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
    """
    Centered-ish moving average via reflection padding + convolution.
    """
    if win <= 1:
        return x
    win = ensure_odd(win)
    pad = win // 2
    xpad = np.pad(x, pad, mode="reflect")
    ker = np.ones(win, dtype=float) / float(win)
    return np.convolve(xpad, ker, mode="valid")

def roll_std_centered(x, win):
    """
    Rolling std, same style as movavg (reflect pad).
    """
    if win <= 1:
        return np.zeros_like(x)
    win = ensure_odd(win)
    m1 = movavg(x, win)
    m2 = movavg(x * x, win)
    var = np.maximum(m2 - m1 * m1, 0.0)
    return np.sqrt(var)

def change_signature(filtered, win):
    """
    The 'signature' used to detect where the event is.
    High where the signal changes quickly / noisily.
    """
    f = filtered - np.median(filtered)
    rmean = movavg(f, win)
    rstd  = roll_std_centered(f, win)
    rstd_m = roll_std_centered(rmean, win)

    sig = rstd * rstd_m
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    smax = np.max(sig) if sig.size else 0.0
    return (sig / smax) if smax > 0 else sig

def center_index_from_time(time, center_time=CENTER_TIME_HINT):
    """
    Pick an index near center_time (usually 0).
    """
    if len(time) == 0:
        return 0
    if time[0] <= center_time <= time[-1]:
        return int(np.argmin(np.abs(time - center_time)))
    return len(time) // 2

def movavg3(x):
    """
    Tiny smoothing for blockage peak estimation.
    """
    x = np.asarray(x, float)
    if x.size < 3:
        return x.copy()
    k = np.array([1.0, 1.0, 1.0]) / 3.0
    xpad = np.pad(x, 1, mode="edge")
    return np.convolve(xpad, k, mode="valid")

def detect_bounds_signature_centered(filtered, time,
                                     win=ROLL_WIN,
                                     peak_height_mult=PEAK_HEIGHT_MULT,
                                     rel_height=REL_HEIGHT,
                                     guard=GUARD_SAMPLES,
                                     min_span=MIN_SPAN,
                                     center_time=CENTER_TIME_HINT,
                                     use_peak_tops=USE_PEAK_TOPS,
                                     peak_guard=GUARD_SAMPLES):
    """
    Uses 'signature' to guess which sample range is the main event.
    Returns (start_index, end_index, central_idx, signature_array, left_peak_idx, right_peak_idx)
    """
    sig = change_signature(filtered, win=win)
    cidx = center_index_from_time(time, center_time=center_time)

    if sig.size == 0:
        left = max(0, cidx - max(min_span // 2, 1))
        right = min(len(filtered) - 1, cidx + max(min_span // 2, 1))
        return left, right, cidx, sig, cidx, cidx

    height = peak_height_mult * np.median(sig)
    distance = max(5, win // 2)
    peaks, _ = find_peaks(sig, height=height, distance=distance, prominence=0.05)

    # fallback if no peaks
    if peaks.size == 0:
        left = max(0, cidx - max(min_span // 2, 1))
        right = min(len(filtered) - 1, cidx + max(min_span // 2, 1))
        return left, right, cidx, sig, cidx, cidx

    left_candidates  = peaks[peaks <= cidx]
    right_candidates = peaks[peaks >= cidx]

    # fallback if we can't bracket
    if left_candidates.size == 0 or right_candidates.size == 0:
        pk = int(peaks[np.argmin(np.abs(peaks - cidx))])
        widths, _, L, R = peak_widths(sig, [pk], rel_height=rel_height)
        left  = max(0, int(np.floor(L[0])) - guard)
        right = min(len(filtered) - 1, int(np.ceil(R[0])) + guard)
        if (right - left) < min_span:
            half = max(min_span // 2, 1)
            left  = max(0, pk - half)
            right = min(len(filtered) - 1, pk + half)
        return left, right, pk, sig, pk, pk

    # we got both sides
    left_peak  = int(left_candidates[np.argmax(left_candidates)])   # furthest-right on left side
    right_peak = int(right_candidates[np.argmin(right_candidates)]) # furthest-left on right side

    if use_peak_tops:
        left  = max(0, left_peak  - max(peak_guard, guard))
        right = min(len(filtered) - 1, right_peak + max(peak_guard, guard))
    else:
        wL, _, L_ips, _ = peak_widths(sig, [left_peak],  rel_height=rel_height)
        wR, _, _, R_ips = peak_widths(sig, [right_peak], rel_height=rel_height)
        left  = max(0, int(np.floor(L_ips[0])) - guard)
        right = min(len(filtered) - 1, int(np.ceil(R_ips[0])) + guard)

    if (right - left) < min_span:
        mid  = (left + right) // 2
        half = max(min_span // 2, 1)
        left  = max(0, mid - half)
        right = min(len(filtered) - 1, mid + half)

    central_ref = int((left_peak + right_peak) // 2)
    return left, right, central_ref, sig, left_peak, right_peak

def real_baseline_from_raw(raw, start, end,
                           guard_pre=200, guard_post=200,
                           trim_q=0.10, min_samples=20,
                           mode=BASELINE_MODE):
    """
    Estimate baseline around the event on the RAW current.
    We look some samples before and after the event window.
    """
    n = len(raw)
    i0, i1 = max(0, start - guard_pre), start
    j0, j1 = end + 1, min(n, end + 1 + guard_post)

    pre  = raw[i0:i1].astype(float, copy=False)
    post = raw[j0:j1].astype(float, copy=False)

    # ensure minimum samples by expanding outwards
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

    drift  = (b_post - b_pre) if np.isfinite(b_pre) and np.isfinite(b_post) else np.nan

    # pooled event baseline
    if mode == "average":
        b_evt = float(np.nanmean([b_pre, b_post]))
    elif mode == "linear" and np.isfinite(b_pre) and np.isfinite(b_post):
        # fit a line to pre+post to get "baseline at event midpoint"
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

    return b_pre, b_post, b_evt, float(drift)

def compute_auc_abs(raw, start, end, dt, baseline, deficit_only=DEFICIT_ONLY):
    """
    AUC of |raw - baseline| over the event window.
    If baseline is scalar => flat line at that level.
    If baseline is array => pointwise baseline over that window.
    """
    seg = raw[start:end+1].astype(float)
    if np.isscalar(baseline):
        dev = seg - float(baseline)
    else:
        base_arr = np.asarray(baseline, float)
        if base_arr.shape[0] != seg.shape[0]:
            raise ValueError("baseline array must match event length")
        dev = seg - base_arr
    if deficit_only:
        # only integrate downward excursions
        return float(np.trapz(np.maximum(-dev, 0.0), dx=dt))
    return float(np.trapz(np.abs(dev), dx=dt))

def linear_map_and_predict(x, y, start, end):
    """
    Fit y ≈ a*x + b on the event window [start:end], then
    yhat = a*x_full + b on the whole trace.
    Returns (a, b, yhat_full)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    seg = slice(start, end+1)
    xs, ys = x[seg], y[seg]

    # If x is flat or too short, fall back to constant baseline
    if xs.size == 0 or np.allclose(np.std(xs), 0.0, atol=1e-12):
        a, b = 0.0, float(np.median(ys)) if ys.size else 0.0
        yhat = np.full_like(y, b, dtype=float)
        return a, b, yhat

    A = np.vstack([xs, np.ones_like(xs)]).T
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(a), float(b), (a * x + b)

def delta_metrics(y_true, y_hat, dt, start, end):
    """
    On [start:end], compute:
        area = integral(|error|)
        mae  = mean absolute error
    """
    seg = slice(start, end+1)
    diff = y_true[seg].astype(float) - y_hat[seg].astype(float)
    if diff.size == 0:
        return float("nan"), float("nan")
    mae = float(np.mean(np.abs(diff)))
    area = float(np.trapz(np.abs(diff), dx=dt))
    return area, mae

def skew_kurtosis_fisher(x):
    """
    Return (skewness, excess_kurtosis) of the baseline-corrected raw segment.
    """
    x = np.asarray(x, float)
    if x.size < 3 or not np.any(np.isfinite(x)):
        return float("nan"), float("nan")
    g1 = skew(x, bias=False, nan_policy="omit")
    g2 = kurtosis(x, fisher=True, bias=False, nan_policy="omit")  # excess kurtosis
    return float(g1), float(g2)

def zone_slices(n, start_idx, end_idx, k):
    """
    Return slice() for pre and post zones of length up to k each.
    pre:  [max(0, start-k) : start)
    post: (end : min(n, end+1+k)]
    """
    pre_start  = max(0, start_idx - k)
    pre_end    = start_idx           # exclusive
    post_start = end_idx + 1
    post_end   = min(n, end_idx + 1 + k)
    pre_sl  = slice(pre_start, pre_end)
    post_sl = slice(post_start, post_end)
    return pre_sl, post_sl

def delta_area_mae_in_zone(raw, curve, dt, sl):
    """
    |raw - curve| area and MAE on the given slice 'sl'.
    Returns (area, mae, n_used).
    """
    seg_r = raw[sl].astype(float)
    seg_c = curve[sl].astype(float)
    n = int(max(0, seg_r.size))
    if n <= 0:
        return float("nan"), float("nan"), 0
    diff = seg_r - seg_c
    area = float(np.trapz(np.abs(diff), dx=dt)) if n > 1 else float(np.abs(diff).sum()) * dt
    mae  = float(np.mean(np.abs(diff)))
    return area, mae, n

# ===================== PLOTTING =====================

def save_plot_isolation(out_png, time_s, raw, filtered,
                        start, end, signature,
                        zoom=True):
    """
    3-panel plot:
      (1) filtered w/ event span
      (2) raw w/ event span
      (3) signature
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios":[2,2,1]}
    )

    # filtered
    axes[0].plot(time_s, filtered, label="Filtered", color="C0")
    axes[0].axvspan(time_s[start], time_s[end], color="orange", alpha=0.25, label="Event")
    axes[0].set_ylabel("Filtered (a.u.)")
    axes[0].legend(loc="upper right")

    # raw
    axes[1].plot(time_s, raw, label="Raw", color="C2")
    axes[1].axvspan(time_s[start], time_s[end], color="orange", alpha=0.25)
    axes[1].set_ylabel("Raw (units)")
    axes[1].legend(loc="upper right")

    # signature
    axes[2].plot(time_s, signature, color="C4", label="Change signature")
    axes[2].axvspan(time_s[start], time_s[end], color="orange", alpha=0.25)
    axes[2].set_ylabel("Sig (norm.)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")

    if zoom:
        tmin = float(time_s[start]) - ZOOM_PAD_SEC
        tmax = float(time_s[end])   + ZOOM_PAD_SEC
        for ax in axes:
            ax.set_xlim(tmin, tmax)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def save_plot_auc(out_png, time_s, raw, filtered,
                  start, end, signature,
                  b_pre, b_post, b_evt, auc_abs,
                  auc_mode=AUC_MODE,
                  zoom=True):
    """
    Plot with AUC shading in RAW panel.
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios":[2,2,1]}
    )

    # filtered
    axes[0].plot(time_s, filtered, label="Filtered", color="C0")
    axes[0].axvspan(time_s[start], time_s[end], color="orange", alpha=0.25, label="Event")
    axes[0].set_ylabel("Filtered (a.u.)")
    axes[0].legend(loc="upper right")

    # raw + baseline shading
    axes[1].plot(time_s, raw, label="Raw", color="C2")

    if auc_mode == "linear":
        base = np.linspace(b_pre, b_post, end - start + 1)
        axes[1].plot(time_s[start:end+1], base, color="k", lw=1.2, ls="--",
                     label="Baseline (linear)")
        y1 = raw[start:end+1]
        y2 = base
    else:
        axes[1].axhline(b_evt, color="k", lw=1.2, ls="--", label="Baseline (flat)")
        y1 = raw[start:end+1]
        y2 = np.full_like(y1, b_evt, dtype=float)

    axes[1].fill_between(
        time_s[start:end+1],
        y1, y2,
        alpha=0.25, color="C3",
        label=f"AUC abs = {auc_abs:.3g}"
    )

    axes[1].axvspan(time_s[start], time_s[end], color="orange", alpha=0.20)
    axes[1].set_ylabel("Raw (units)")
    axes[1].legend(loc="upper right")

    # signature
    axes[2].plot(time_s, signature, color="C4", label="Change signature")
    axes[2].axvspan(time_s[start], time_s[end], color="orange", alpha=0.25)
    axes[2].set_ylabel("Sig (norm.)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")

    if zoom:
        tmin = float(time_s[start]) - ZOOM_PAD_SEC
        tmax = float(time_s[end])   + ZOOM_PAD_SEC
        for ax in axes:
            ax.set_xlim(tmin, tmax)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_delta(out_png, time_s, raw, curve, label_curve,
               start, end, zoom=True, note=""):
    """
    Compare raw vs mapped curve (filtered or fitted mapped to raw units),
    and shade |Δ|.
    """
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(time_s, raw,   label="Raw",          color="C2", lw=1.2)
    ax.plot(time_s, curve, label=label_curve,    color="C0", lw=1.2)

    start = int(max(0, min(start, len(time_s)-1)))
    end   = int(max(start, min(end, len(time_s)-1)))

    ax.fill_between(time_s[start:end+1],
                    raw[start:end+1],
                    curve[start:end+1],
                    color="C3", alpha=0.25,
                    label="|Δ| area")

    ax.axvspan(time_s[start], time_s[end], color="orange", alpha=0.2, zorder=0)

    if note:
        ax.set_title(note)
    ax.set_ylabel("Current (raw units)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")

    if zoom:
        tmin = float(time_s[start]) - ZOOM_PAD_SEC
        tmax = float(time_s[end])   + ZOOM_PAD_SEC
        ax.set_xlim(tmin, tmax)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140)
    plt.close(fig)

def plot_noise_overlay(
    out_png, time_s, raw, filt_mapped, fit_mapped,
    start_idx, end_idx, pre_sl, post_sl,
    noise_area_filt, noise_mae_filt,
    noise_area_fit,  noise_mae_fit
):
    """
    Show raw + mapped filtered (+ optional mapped fit),
    highlight event (orange) and pre/post zones (purple),
    and annotate merged noise metrics.
    """
    fig, ax = plt.subplots(figsize=(10.5, 4.4))

    # raw and mappings
    ax.plot(time_s, raw, label="Raw", lw=1.0)
    ax.plot(time_s, filt_mapped, label="Mapped filtered", lw=1.0)

    # optional fit mapping
    if fit_mapped is not None and fit_mapped.size == raw.size and np.any(np.isfinite(fit_mapped)):
        ax.plot(time_s, fit_mapped, label="Mapped fit", lw=1.0)

    # event window
    ax.axvspan(time_s[start_idx], time_s[end_idx], color="orange", alpha=0.20, zorder=0, label="Event")

    # pre/post zones
    if pre_sl.stop - pre_sl.start > 0:
        ax.axvspan(time_s[pre_sl.start], time_s[pre_sl.stop-1], color="purple", alpha=0.12, zorder=0, label="Pre zone")
    if post_sl.stop - post_sl.start > 0:
        ax.axvspan(time_s[post_sl.start], time_s[post_sl.stop-1], color="purple", alpha=0.12, zorder=0, label="Post zone")

    # annotation text
    txt = [
        f"Noise Δ(filtered): area={noise_area_filt:.3g}, MAE={noise_mae_filt:.3g}"
    ]
    if np.isfinite(noise_area_fit) or np.isfinite(noise_mae_fit):
        txt.append(f"Noise Δ(fit): area={noise_area_fit:.3g}, MAE={noise_mae_fit:.3g}")
    ax.text(0.01, 0.98, "\n".join(txt), transform=ax.transAxes, va="top", ha="left", fontsize=9)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (raw units)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


# ===================== CORE =====================

def load_fits(fit_json_path):
    """
    Load peak_fits/fit_results.json into a dict keyed by peak_index.
    Handles list or dict formats.
    """
    with open(fit_json_path, "r") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = list(data.values())
    else:
        rows = []

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
    """
    Process one sample's fit_results.json and produce:
      - event_features_with_deltas.csv
      - plots/
    """
    sample_dir = os.path.dirname(os.path.dirname(fit_json_path))
    print("Processing fits-only sample:", sample_dir)

    # Load all peaks for this sample
    fit_map = load_fits(fit_json_path)
    if not fit_map:
        print("  (no peaks in this fit file)")
        return

    # Prepare output paths
    out_csv         = os.path.join(sample_dir, "event_features_with_deltas.csv")
    plot_root       = os.path.join(sample_dir, "plots")
    dir_iso         = os.path.join(plot_root, "event_isolation")
    dir_auc         = os.path.join(plot_root, "auc")
    dir_delta_filt  = os.path.join(plot_root, "delta_filtered")
    dir_delta_fit   = os.path.join(plot_root, "delta_fit")
    for d in [dir_iso, dir_auc, dir_delta_filt, dir_delta_fit]:
        os.makedirs(d, exist_ok=True)

    # create CSV with header (overwrite each run)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "fit_json","event_idx",
            "start_idx","end_idx",
            "t_start_s","t_end_s","duration_s",
            "peak_center_idx","roll_win","rel_height","height_mult",
            "raw_baseline_pre","raw_baseline_post","raw_baseline_event","raw_drift_pp",
            "auc_abs",
            "average_blockage","maximum_blockage",
            "delta_area_filtered_raw","delta_mae_filtered_raw",
            "delta_area_fit_raw","delta_mae_fit_raw",
            "map_a_filtered","map_b_filtered",
            "map_a_fit","map_b_fit",
            "fit_available",
            "skewness_raw_resid","kurtosis_excess_raw_resid",
            # --- NEW merged noise features (pre+post) ---
            "noise_n",
            "noise_delta_area_filtered_raw","noise_delta_mae_filtered_raw",
            "noise_delta_area_fit_raw","noise_delta_mae_fit_raw"
        ])

    # plot counters to limit spam
    n_iso = n_auc = n_df = n_dfit = 0

    # iterate peaks
    for i in sorted(fit_map.keys()):
        try:
            payload = fit_map[i]
            tr = payload.get("trace", {})

            # --- pull arrays from trace ---
            # t is in µs in new pipeline → convert to seconds immediately
            t_us       = np.array(tr.get("t", []), dtype=float)
            raw_full   = np.array(tr.get("raw", []), dtype=float)
            filt_full  = np.array(tr.get("filtered", []), dtype=float)
            fit_full   = np.array(tr.get("fitted", []), dtype=float)

            if not (t_us.size and raw_full.size and filt_full.size):
                print(f"  peak {i}: missing arrays, skipping")
                continue

            n = min(t_us.size, raw_full.size, filt_full.size)
            if fit_full.size:
                n = min(n, fit_full.size)

            t_us     = t_us[:n]
            raw_full = raw_full[:n]
            filt_full= filt_full[:n]
            if fit_full.size:
                fit_full = fit_full[:n]

            # convert µs -> s for downstream math/plots
            t_s = t_us * 1e-6

            # compute dt in seconds
            if n >= 2:
                dt_s = float(np.median(np.diff(t_s)))
            else:
                dt_s = 1.0  # fallback

            # --- detect event window on filtered ---
            start_idx, end_idx, peak_center_idx, sig_arr, _, _ = detect_bounds_signature_centered(
                filt_full, t_s,
                win=ROLL_WIN,
                peak_height_mult=PEAK_HEIGHT_MULT,
                rel_height=REL_HEIGHT,
                guard=GUARD_SAMPLES,
                min_span=MIN_SPAN,
                center_time=CENTER_TIME_HINT,
                use_peak_tops=USE_PEAK_TOPS,
                peak_guard=GUARD_SAMPLES
            )

            # clamp indices to safe range
            start_idx = int(max(0, min(start_idx, n-1)))
            end_idx   = int(max(start_idx, min(end_idx, n-1)))

            # --- baselines from RAW ---
            b_pre, b_post, b_evt, drift_pp = real_baseline_from_raw(
                raw_full, start_idx, end_idx,
                guard_pre=200, guard_post=200,
                trim_q=0.10, min_samples=20,
                mode=BASELINE_MODE
            )

            # --- event AUC on RAW ---
            if AUC_MODE == "linear":
                baseline_for_auc = np.linspace(b_pre, b_post, end_idx - start_idx + 1)
            else:
                baseline_for_auc = b_evt
            auc_abs = compute_auc_abs(
                raw_full, start_idx, end_idx,
                dt_s, baseline_for_auc,
                deficit_only=DEFICIT_ONLY
            )

            # --- blockage metrics ---
            seg_raw = raw_full[start_idx:end_idx+1].astype(float)
            if AUC_MODE == "linear":
                base_arr = np.linspace(b_pre, b_post, end_idx - start_idx + 1)
                deficit  = base_arr - seg_raw
            else:
                deficit  = b_evt - seg_raw

            duration_s = float((end_idx - start_idx) * dt_s)
            if duration_s > 0:
                average_blockage = float(auc_abs / duration_s)
            else:
                average_blockage = float("nan")

            if deficit.size:
                peak_def_smooth = float(np.max(movavg3(np.abs(deficit))))
            else:
                peak_def_smooth = float("nan")

            if peak_def_smooth > 0 and np.isfinite(peak_def_smooth):
                maximum_blockage = float(auc_abs / peak_def_smooth)
            else:
                maximum_blockage = float("nan")

            # --- map filtered -> raw (physical units) ---
            a_f, b_f, filt_mapped = linear_map_and_predict(
                filt_full, raw_full,
                start_idx, end_idx
            )
            d_area_fr, d_mae_fr = delta_metrics(
                raw_full, filt_mapped,
                dt_s, start_idx, end_idx
            )

            # --- map fitted -> raw (handle NaNs) ---
            fit_available = False
            a_fit = b_fit = np.nan
            d_area_fit = d_mae_fit = np.nan
            fit_mapped = None

            if fit_full.size:
                fit_ok = fit_full.copy()
                if np.any(~np.isfinite(fit_ok)):
                    valid = np.where(np.isfinite(fit_ok))[0]
                    if valid.size:
                        first_valid = valid[0]
                        last_valid  = valid[-1]
                        # fill leading/trailing
                        for k in range(0, first_valid):
                            fit_ok[k] = fit_ok[first_valid]
                        for k in range(last_valid+1, fit_ok.size):
                            fit_ok[k] = fit_ok[last_valid]
                        # fill internal NaNs linearly
                        bad = np.where(~np.isfinite(fit_ok))[0]
                        if bad.size:
                            good = np.where(np.isfinite(fit_ok))[0]
                            fit_ok[bad] = np.interp(bad, good, fit_ok[good])
                    else:
                        fit_ok[:] = 0.0

                a_fit, b_fit, fit_mapped = linear_map_and_predict(
                    fit_ok, raw_full,
                    start_idx, end_idx
                )
                d_area_fit, d_mae_fit = delta_metrics(
                    raw_full, fit_mapped,
                    dt_s, start_idx, end_idx
                )
                fit_available = True

            # --- shape stats on residual raw segment ---
            if AUC_MODE == "linear":
                base_for_mom = np.linspace(b_pre, b_post, end_idx - start_idx + 1)
            else:
                base_for_mom = np.full(end_idx - start_idx + 1, b_evt, dtype=float)

            resid = seg_raw - base_for_mom
            skew_raw, kurt_ex_raw = skew_kurtosis_fisher(resid)

            # ---------- NEW: merged pre+post noise contribution ----------
            noise_n = 0
            noise_delta_area_filtered_raw = float("nan")
            noise_delta_mae_filtered_raw  = float("nan")
            noise_delta_area_fit_raw      = float("nan")
            noise_delta_mae_fit_raw       = float("nan")

            # build pre/post slices
            pre_sl, post_sl = zone_slices(n, start_idx, end_idx, PRE_POST_SAMPLES)

            # FILTERED→RAW noise (merged)
            pre_darea_fr, pre_dmae_fr, pre_n  = delta_area_mae_in_zone(raw_full, filt_mapped, dt_s, pre_sl)
            post_darea_fr, post_dmae_fr, post_n = delta_area_mae_in_zone(raw_full, filt_mapped, dt_s, post_sl)

            if pre_n  < PRE_POST_MIN_SAMPLES:
                pre_darea_fr = float("nan"); pre_dmae_fr = float("nan"); pre_n = 0
            if post_n < PRE_POST_MIN_SAMPLES:
                post_darea_fr = float("nan"); post_dmae_fr = float("nan"); post_n = 0

            noise_n = pre_n + post_n
            if noise_n > 0:
                # areas add; MAE is sample-weighted
                area_parts = [x for x in (pre_darea_fr, post_darea_fr) if np.isfinite(x)]
                noise_delta_area_filtered_raw = float(np.nansum(area_parts)) if area_parts else float("nan")
                num = 0.0
                if pre_n  > 0 and np.isfinite(pre_dmae_fr):  num += pre_dmae_fr * pre_n
                if post_n > 0 and np.isfinite(post_dmae_fr): num += post_dmae_fr * post_n
                noise_delta_mae_filtered_raw = float(num / noise_n) if noise_n > 0 else float("nan")

            # FIT→RAW noise (merged) if available
            if fit_available and fit_mapped is not None and fit_mapped.size == n:
                pre_darea_fit, pre_dmae_fit, pre_n_fit   = delta_area_mae_in_zone(raw_full, fit_mapped, dt_s, pre_sl)
                post_darea_fit, post_dmae_fit, post_n_fit = delta_area_mae_in_zone(raw_full, fit_mapped, dt_s, post_sl)

                if pre_n_fit  < PRE_POST_MIN_SAMPLES:
                    pre_darea_fit = float("nan"); pre_dmae_fit = float("nan"); pre_n_fit = 0
                if post_n_fit < PRE_POST_MIN_SAMPLES:
                    post_darea_fit = float("nan"); post_dmae_fit = float("nan"); post_n_fit = 0

                noise_n_fit = pre_n_fit + post_n_fit
                if noise_n_fit > 0:
                    area_parts_fit = [x for x in (pre_darea_fit, post_darea_fit) if np.isfinite(x)]
                    noise_delta_area_fit_raw = float(np.nansum(area_parts_fit)) if area_parts_fit else float("nan")
                    num_fit = 0.0
                    if pre_n_fit  > 0 and np.isfinite(pre_dmae_fit):   num_fit += pre_dmae_fit * pre_n_fit
                    if post_n_fit > 0 and np.isfinite(post_dmae_fit):  num_fit += post_dmae_fit * post_n_fit
                    noise_delta_mae_fit_raw = float(num_fit / noise_n_fit) if noise_n_fit > 0 else float("nan")

            # --- write row to CSV ---
            t_start_s = float(t_s[start_idx])
            t_end_s   = float(t_s[end_idx])

            with open(out_csv, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    os.path.basename(fit_json_path), i,
                    start_idx, end_idx,
                    t_start_s, t_end_s, duration_s,
                    int(peak_center_idx),
                    ROLL_WIN, REL_HEIGHT, PEAK_HEIGHT_MULT,
                    b_pre, b_post, b_evt, drift_pp,
                    auc_abs,
                    average_blockage, maximum_blockage,
                    d_area_fr, d_mae_fr,
                    d_area_fit, d_mae_fit,
                    a_f, b_f,
                    a_fit, b_fit,
                    int(fit_available),
                    skew_raw, kurt_ex_raw,
                    int(noise_n),
                    noise_delta_area_filtered_raw, noise_delta_mae_filtered_raw,
                    noise_delta_area_fit_raw,      noise_delta_mae_fit_raw
                ])

            # --- standard plots (rate-limited) ---
            make_plot = (i % PLOT_EVERY == 0)

            if make_plot and SAVE_PLOT_ISO and n_iso < MAX_PLOTS_PER_FOLDER:
                out_iso = os.path.join(dir_iso, f"event_{i:05d}.png")
                save_plot_isolation(
                    out_iso,
                    t_s, raw_full, filt_full,
                    start_idx, end_idx, sig_arr,
                    zoom=ZOOM_MODE
                )
                n_iso += 1

            if make_plot and SAVE_PLOT_AUC and n_auc < MAX_PLOTS_PER_FOLDER:
                out_auc = os.path.join(dir_auc, f"event_{i:05d}.png")
                save_plot_auc(
                    out_auc,
                    t_s, raw_full, filt_full,
                    start_idx, end_idx, sig_arr,
                    b_pre, b_post, b_evt,
                    auc_abs,
                    auc_mode=AUC_MODE,
                    zoom=ZOOM_MODE
                )
                n_auc += 1

            if make_plot and SAVE_PLOT_DELTA_FILT and n_df < MAX_PLOTS_PER_FOLDER:
                out_df = os.path.join(dir_delta_filt, f"event_{i:05d}.png")
                note_fr = (
                    f"Δ(filtered, raw) in-event: area={d_area_fr:.3g}, "
                    f"MAE={d_mae_fr:.3g}"
                )
                plot_delta(
                    out_df,
                    t_s, raw_full, filt_mapped,
                    "Mapped filtered",
                    start_idx, end_idx,
                    zoom=ZOOM_MODE,
                    note=note_fr
                )
                n_df += 1

            if (make_plot and SAVE_PLOT_DELTA_FIT and fit_available
                    and n_dfit < MAX_PLOTS_PER_FOLDER):
                out_dfit = os.path.join(dir_delta_fit, f"event_{i:05d}.png")
                note_fit = (
                    f"Δ(fit, raw) in-event: area={d_area_fit:.3g}, "
                    f"MAE={d_mae_fit:.3g}"
                )
                plot_delta(
                    out_dfit,
                    t_s, raw_full, fit_mapped,
                    "Mapped fit",
                    start_idx, end_idx,
                    zoom=ZOOM_MODE,
                    note=note_fit
                )
                n_dfit += 1

            # Optional QA plots for pre/post noise zones
            if EVAL_PRE_POST_PLOTS and make_plot:
                os.makedirs(os.path.join(plot_root, "delta_prepost"), exist_ok=True)
                # Pre zone (filtered map)
                if PRE_POST_SAMPLES > 0 and (pre_sl.stop - pre_sl.start) > 0:
                    out_pre = os.path.join(plot_root, "delta_prepost", f"event_{i:05d}_pre_filt.png")
                    plot_delta(out_pre, t_s, raw_full, filt_mapped, "Mapped filtered",
                               pre_sl.start, max(pre_sl.start, pre_sl.stop-1),
                               zoom=False,
                               note="PRE zone (filtered map)")
                # Post zone (filtered map)
                if PRE_POST_SAMPLES > 0 and (post_sl.stop - post_sl.start) > 0:
                    out_post = os.path.join(plot_root, "delta_prepost", f"event_{i:05d}_post_filt.png")
                    plot_delta(out_post, t_s, raw_full, filt_mapped, "Mapped filtered",
                               post_sl.start, max(post_sl.start, post_sl.stop-1),
                               zoom=False,
                               note="POST zone (filtered map)")
                    
                if EVAL_PRE_POST_PLOTS and (i % PLOT_EVERY == 0):
                    dir_noise = os.path.join(plot_root, "noise_overlay")
                    os.makedirs(dir_noise, exist_ok=True)
                    out_noise = os.path.join(dir_noise, f"event_{i:05d}_noise.png")
                    fit_map_for_plot = fit_mapped if (fit_available and fit_mapped is not None and fit_mapped.size == n) else None
                    plot_noise_overlay(
                        out_noise, t_s, raw_full, filt_mapped, fit_map_for_plot,
                        start_idx, end_idx, pre_sl, post_sl,
                        noise_delta_area_filtered_raw, noise_delta_mae_filtered_raw,
                        noise_delta_area_fit_raw,      noise_delta_mae_fit_raw
                    )

        except Exception as e:
            print(f"  peak {i}: ERROR -> {e}")

    # ================== SAMPLE-LEVEL SUMMARY PLOTS ==================
    try:
        import csv as _csv

        with open(out_csv, "r") as _fh:
            rdr = _csv.reader(_fh)
            header = next(rdr)
            idx_map = {name: idx for idx, name in enumerate(header)}
            # grab arrays we want to summarize
            arr_noise_area_filt = []
            arr_noise_mae_filt  = []
            arr_noise_area_fit  = []
            arr_noise_mae_fit   = []
            arr_delta_area_evt_filt = []
            arr_delta_area_evt_fit  = []
            arr_auc_abs = []
            arr_event_idx = []

            for row in rdr:
                def fget(name):
                    try:
                        v = row[idx_map[name]]
                        return float(v) if v not in ("", "nan", "NaN", "None") else np.nan
                    except Exception:
                        return np.nan

                arr_noise_area_filt.append( fget("noise_delta_area_filtered_raw") )
                arr_noise_mae_filt.append(  fget("noise_delta_mae_filtered_raw") )
                arr_noise_area_fit.append(  fget("noise_delta_area_fit_raw") )
                arr_noise_mae_fit.append(   fget("noise_delta_mae_fit_raw") )
                arr_delta_area_evt_filt.append( fget("delta_area_filtered_raw") )
                arr_delta_area_evt_fit.append(  fget("delta_area_fit_raw") )
                arr_auc_abs.append( fget("auc_abs") )
                # event index (int)
                try:
                    arr_event_idx.append( int(float(row[idx_map["event_idx"]])) )
                except Exception:
                    arr_event_idx.append( np.nan )

        arr_noise_area_filt = np.array(arr_noise_area_filt, float)
        arr_noise_mae_filt  = np.array(arr_noise_mae_filt, float)
        arr_noise_area_fit  = np.array(arr_noise_area_fit, float)
        arr_noise_mae_fit   = np.array(arr_noise_mae_fit, float)
        arr_delta_area_evt_filt = np.array(arr_delta_area_evt_filt, float)
        arr_delta_area_evt_fit  = np.array(arr_delta_area_evt_fit, float)
        arr_auc_abs = np.array(arr_auc_abs, float)
        arr_event_idx = np.array(arr_event_idx, float)

        # Output dir
        sum_dir = os.path.join(plot_root, "noise_summary")
        os.makedirs(sum_dir, exist_ok=True)

        # (A) Histograms of noise metrics
        def _simple_hist(x, title, xlabel, fname, bins=40):
            x = x[np.isfinite(x)]
            if x.size == 0: return
            plt.figure(figsize=(6.5,4.0))
            plt.hist(x, bins=bins, edgecolor="black")
            plt.title(title)
            plt.xlabel(xlabel); plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(sum_dir, fname), dpi=150)
            plt.close()

        _simple_hist(arr_noise_area_filt, "Noise |Δ(filtered→raw)| area", "area (raw·s)", "hist_noise_area_filtered.png")
        _simple_hist(arr_noise_mae_filt,  "Noise MAE (filtered→raw)",     "MAE (raw units)", "hist_noise_mae_filtered.png")

        if np.any(np.isfinite(arr_noise_area_fit)):
            _simple_hist(arr_noise_area_fit, "Noise |Δ(fit→raw)| area", "area (raw·s)", "hist_noise_area_fit.png")
        if np.any(np.isfinite(arr_noise_mae_fit)):
            _simple_hist(arr_noise_mae_fit,  "Noise MAE (fit→raw)",     "MAE (raw units)", "hist_noise_mae_fit.png")

        # (B) Scatter: event Δ vs noise Δ  (does noise predict in-event residuals?)
        def _scatter(x, y, title, xlabel, ylabel, fname):
            m = np.isfinite(x) & np.isfinite(y)
            if not np.any(m): return
            plt.figure(figsize=(5.6,4.6))
            plt.scatter(x[m], y[m], alpha=0.6, s=16)
            plt.title(title)
            plt.xlabel(xlabel); plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(os.path.join(sum_dir, fname), dpi=150)
            plt.close()

        _scatter(arr_noise_area_filt, arr_delta_area_evt_filt,
                "Event Δ-area vs Noise Δ-area (filtered map)",
                "Noise |Δ(filtered→raw)| area", "In-event |Δ(filtered→raw)| area",
                "scatter_noise_vs_event_filtered_area.png")

        if np.any(np.isfinite(arr_noise_area_fit)) and np.any(np.isfinite(arr_delta_area_evt_fit)):
            _scatter(arr_noise_area_fit, arr_delta_area_evt_fit,
                    "Event Δ-area vs Noise Δ-area (fit map)",
                    "Noise |Δ(fit→raw)| area", "In-event |Δ(fit→raw)| area",
                    "scatter_noise_vs_event_fit_area.png")

        # (C) Drift over event index: does noise grow across the sample?
        def _line(x_idx, y, title, ylabel, fname):
            m = np.isfinite(x_idx) & np.isfinite(y)
            if np.sum(m) < 3: return
            order = np.argsort(x_idx[m])
            plt.figure(figsize=(6.8,4.0))
            plt.plot(x_idx[m][order], y[m][order], lw=1.0, marker=".", ms=3)
            plt.title(title)
            plt.xlabel("Event index (sorted as processed)"); plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(os.path.join(sum_dir, fname), dpi=150)
            plt.close()

        _line(arr_event_idx, arr_noise_mae_filt, "Noise MAE (filtered map) vs event index", "MAE (raw units)", "line_noise_mae_filtered_vs_idx.png")
        if np.any(np.isfinite(arr_noise_mae_fit)):
            _line(arr_event_idx, arr_noise_mae_fit, "Noise MAE (fit map) vs event index", "MAE (raw units)", "line_noise_mae_fit_vs_idx.png")

    except Exception as _e:
        print("  (summary plots skipped due to error:", _e, ")")


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

if __name__ == "__main__":
    __main__()
