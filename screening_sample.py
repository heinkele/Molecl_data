# =========================
# ========= CONFIG ========
# =========================
CONFIG = {
    # ---- Paths / IO ----
    "ROOT_PATH": "/Users/hugo/New data/PacBio",   # where sample1/, sample2/, ...
    "FILE_GLOB": ["*.dat", "*.abf"],              # support both formats
    "SKIP_ALREADY_PROCESSED": True,               # uses processing_status.json

    # ---- Header parsing (.dat only) ----
    "HEADER_SAMPLERATE_DIV": 1,

    # ---- Baseline estimation ----
    # Alexander uses 1 kHz for normalization (before segmentation)
    "BASELINE_LP_HZ": 1e3,
    "BASELINE_ORDER": 4,

    # ---- Segmentation ----
    # Same normalized gradient threshold and min segment length as Alexander
    "SEG_GRAD_THRESH": 0.3,
    "MIN_SEG_LEN": 50000,
    "SEG_VAR_MAX": 5e-3,         # reject segments with higher normalized variance
    "SEGMENT_LP_HZ": None,       # no extra smoothing before segmentation

    # ---- Feature extraction ----
    # Alexander uses ratio mode: 1 - I/I_norm
    "FEATURE_MODE": "ratio",
    "EPS_DENOM": 1e-12,

    # Low-pass filter on the feature signal
    "FEATURE_LP_ENABLE": True,
    "FEATURE_LP_MODE": "fixed_hz",   # use fixed cutoff (not fraction of fs)
    "FEATURE_LP_FIXED_HZ": 5e5,      # 500 kHz cutoff for feature LP
    "FEATURE_LP_FRAC": 0.20,         # unused unless FEATURE_LP_MODE == "frac_fs"
    "FEATURE_LP_ORDER": 6,

    # ---- Peak detection ----
    "WINDOW_SIZE": 2500,          # samples
    "PEAK_PROMINENCE": 0.2,       # stronger peaks only
    "PEAK_MIN_DISTANCE": 1000,    # samples
    "SNR_REPORT": True,           # compute and include SNR in output

    # ---- Plotting / debug ----
    "BG_PLOT_DOWNSAMPLE": 1000,   # for background.png
    "SEG_PLOT_DOWNSAMPLE": 100,   # for kept/rejected segment plots
    "SAVE_SEGMENT_DEBUG_WHEN_REJECTED": True,
    "SAVE_SEGMENT_DEBUG_WHEN_ACCEPTED": True,

    # ---- Safety / numeric ----
    "MIN_BASELINE_MEDIAN": 1e-18,
    "MIN_WINDOW_SAMPLES": 1000,
    "ENFORCE_DISTANCE_GTE_WINDOW": True,
}



# =========================
# ====== IMPLEMENTATION ===
# =========================
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from scipy.signal import savgol_filter, filtfilt, butter, find_peaks


# ---------- tiny helpers ----------
def ensure_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def butter_lowpass_filter(data, cutoff_hz, fs_hz, order=5):
    nyq = fs_hz / 2.0
    cutoff = float(max(1.0, min(cutoff_hz, 0.99 * nyq)))
    b, a = butter(order, cutoff, fs=fs_hz, btype='low', analog=False)
    return filtfilt(b, a, data)

def find_sudden_changes(signal, threshold=0.1, min_segment_length=1000):
    """
    Return [(start_idx, end_idx), ...] for quasi-stable subsegments
    separated by sharp gradient changes.
    """
    grad = np.gradient(signal)
    maxg = np.max(np.abs(grad))
    grad_norm = grad / (maxg if maxg != 0 else 1.0)

    change_points = np.where(np.abs(grad_norm) > threshold)[0]
    all_points = np.concatenate(([0], change_points, [len(signal)-1]))

    segments = []
    for i in range(len(all_points)-1):
        a, b = all_points[i], all_points[i+1]
        if b - a >= min_segment_length:
            segments.append((a, b))
    return segments


# ---------- bookkeeping for processed status ----------
def get_data_folders(root_path):
    folders = []
    for item in os.listdir(root_path):
        if item.startswith('.') or item == '__pycache__':
            continue
        full_path = os.path.join(root_path, item)
        if os.path.isdir(full_path) and 'baseline' not in item.lower():
            folders.append(full_path)
    return sorted(folders)

def _status_file(root_path):
    return os.path.join(root_path, 'processing_status.json')

def load_processing_status(root_path):
    status_file = _status_file(root_path)
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return json.load(f)
    return {}

def save_processing_status(root_path, status):
    status_file = _status_file(root_path)
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

def mark_file_as_processed(root_path, data_folder, file_path):
    status = load_processing_status(root_path)
    status.setdefault(data_folder, [])
    if file_path not in status[data_folder]:
        status[data_folder].append(file_path)
        save_processing_status(root_path, status)

def is_file_processed(root_path, data_folder, file_path):
    status = load_processing_status(root_path)
    return data_folder in status and file_path in status[data_folder]


# =========================
# ===== FILE LOADERS ======
# =========================

# ---- Axopatch .dat loader ----
def _parse_header_bytes_axopatch(header_bytes: bytes, samplerate_div: float
                                 ) -> Tuple[Optional[float], Optional[int], bool]:
    """
    Pull samplerate [Hz], FEMTO LP cutoff if present, and graphene flag.
    """
    txt = header_bytes.decode('latin-1', errors='ignore')
    samplerate = None
    femtoLP = None
    graphene = False
    for line in txt.splitlines():
        if ('Acquisition' in line) or ('Sample Rate' in line):
            digits = ''.join(ch for ch in line if ch.isdigit())
            if digits:
                samplerate = int(digits) / samplerate_div
        if 'FEMTO preamp Bandwidth' in line:
            digits = ''.join(ch for ch in line if ch.isdigit())
            if digits:
                femtoLP = int(digits)
        if 'I_Graphene' in line:
            graphene = True
    return samplerate, femtoLP, graphene

def load_axopatch_dat(file_path: str, cfg: Dict) -> Dict:
    """
    Return:
        {
          "samplerate": fs_Hz (float),
          "I": np.array (A),
          "t": np.array (s),
          "meta": {...},
          "filename": "file.dat"
        }
    """
    x = np.fromfile(file_path, np.dtype('>f4'))  # big-endian float32
    with open(file_path, 'rb') as f:
        header_bytes = f.read(4096)

    sr_div = cfg["HEADER_SAMPLERATE_DIV"]
    samplerate, femtoLP, graphene = _parse_header_bytes_axopatch(header_bytes, sr_div)
    if samplerate is None:
        raise ValueError(f"Could not parse sample rate from header for {file_path}")

    end = len(x)
    if graphene:
        i1 = x[250:end-3:4]
        # i2 = x[251:end-2:4]  # available if you ever need channel2
        # v1 = x[252:end-1:4]
        # v2 = x[253:end:4]
    else:
        i1 = x[250:end-1:2]
        # v1 = x[251:end:2]

    I = np.asarray(i1, dtype=np.float64)
    t = np.arange(I.size, dtype=np.float64) / float(samplerate)

    meta = {
        "type": "AxopatchDAT",
        "graphene": bool(graphene),
        "FemtoLowPass_Hz": femtoLP if femtoLP is not None else None,
    }
    return {
        "samplerate": float(samplerate),
        "I": I,
        "t": t,
        "meta": meta,
        "filename": os.path.basename(file_path),
    }


# ---- ABF loader ----
def load_abf(file_path: str, cfg: Dict) -> Dict:
    """
    Returns a dict with the same format as load_axopatch_dat():
        {
          "samplerate": fs_Hz,
          "I": np.ndarray (current in Amps),
          "t": np.ndarray (time in seconds),
          "meta": {...},
          "filename": "filename.abf"
        }
    """
    import pyabf
    try:
        df = pyabf.ABF(file_path)
    except Exception as e:
        raise RuntimeError(f"Error loading ABF file {file_path}: {e}")

    # Use first sweep
    df.setSweep(0)
    I = np.array(df.sweepY, dtype=np.float64) * 1e-12   # pA -> A
    t = np.array(df.sweepX, dtype=np.float64)           # seconds

    # Compute sampling rate
    fs = 1.0 / abs(t[1] - t[0]) if len(t) > 1 else 1.0

    # Meta information for logging / traceability
    meta = {
        "type": "ABF",
        "abfVersion": df.abfVersionString,
        "sweepCount": int(df.sweepCount),
        "channelName": df.adcNames[0] if df.adcNames else "CH0",
        "units": df.adcUnits[0] if df.adcUnits else "pA",
        "fs_Hz": fs,
    }

    return {
        "samplerate": float(fs),
        "I": I,       # Current in Amps
        "t": t,       # Time in seconds
        "meta": meta,
        "filename": os.path.basename(file_path),
    }



def load_recording(file_path: str, cfg: Dict) -> Dict:
    """
    Simple dispatcher by file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".dat":
        return load_axopatch_dat(file_path, cfg)
    elif ext == ".abf":
        return load_abf(file_path, cfg)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# =========================
# === CORE SCREENING ======
# =========================

def process_one_file(file_path: str, data_path: str, cfg: Dict):
    """
    EXACT same high-level behavior as original script, but:
    - we call load_recording() instead of load_axopatch_dat() directly
    - we still write peaks_data.json
    - we still write background.png
    - we still make kept/discarded segment plots and figures/peak_*.png
    """
    print(f"  Processing: {os.path.basename(file_path)}")

    # Output dirs for this recording
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    output_dir    = os.path.join(data_path, base_filename)
    figures_dir   = os.path.join(output_dir, 'figures')
    kept_dir      = os.path.join(output_dir, 'kept_segments')
    disc_dir      = os.path.join(output_dir, 'discarded_segments')

    ensure_dir(output_dir)
    ensure_dir(figures_dir)
    ensure_dir(kept_dir)
    ensure_dir(disc_dir)

    # ---- load data (format-agnostic) ----
    try:
        rec = load_recording(file_path, cfg)
    except Exception as e:
        print(f"    ERROR loading {os.path.basename(file_path)}: {e}")
        return

    I  = rec["I"].astype(float)              # current [A]
    t  = rec["t"].astype(float)              # time [s]
    fs = float(rec["samplerate"])            # Hz

    # ---- baseline LP ----
    I_norm = butter_lowpass_filter(I, cfg["BASELINE_LP_HZ"], fs, cfg["BASELINE_ORDER"])

    # ---- background plot (same style) ----
    ds_bg   = cfg["BG_PLOT_DOWNSAMPLE"]
    med_b   = np.median(I_norm)
    denom_b = med_b if abs(med_b) > cfg["MIN_BASELINE_MEDIAN"] else cfg["MIN_BASELINE_MEDIAN"]
    variation = float(np.std(I_norm) / denom_b)

    fig_bg, ax_bg = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    ax_bg.plot(t[::ds_bg], I[::ds_bg],      c='tab:blue',   label='Raw I')
    ax_bg.plot(t[::ds_bg], I_norm[::ds_bg], c='tab:orange', label=f'LP baseline ({int(cfg["BASELINE_LP_HZ"])} Hz)')
    ax_bg.set_xlabel('Time (s)')
    ax_bg.set_ylabel('Current (A)')
    ax_bg.set_title(f'Baseline variation: {np.round(variation, 4)}')
    ax_bg.legend()
    fig_bg.tight_layout()
    plt.savefig(os.path.join(output_dir, 'background.png'))
    plt.close(fig_bg)

    # ---- build segmentation source ----
    if cfg["SEGMENT_LP_HZ"] is not None:
        seg_signal = butter_lowpass_filter(I, cfg["SEGMENT_LP_HZ"], fs, order=3)
    else:
        seg_signal = I_norm

    segments = find_sudden_changes(
        seg_signal,
        threshold=cfg["SEG_GRAD_THRESH"],
        min_segment_length=cfg["MIN_SEG_LEN"]
    )
    print(f"    Segments found: {len(segments)}")

    peak_data: List[Dict] = []
    global_peak_counter = 0

    # ---- loop over segments ----
    for seg_start, seg_end in segments:
        I_seg      = I[seg_start:seg_end]
        I_norm_seg = I_norm[seg_start:seg_end]
        t_seg      = t[seg_start:seg_end]

        med_local   = np.median(I_norm_seg)
        denom_local = med_local if abs(med_local) > cfg["MIN_BASELINE_MEDIAN"] else cfg["MIN_BASELINE_MEDIAN"]
        seg_variation = float(np.std(I_norm_seg) / denom_local)

        kept = (seg_variation <= cfg["SEG_VAR_MAX"])

        # segment debug plot(s)
        if kept and cfg.get("SAVE_SEGMENT_DEBUG_WHEN_ACCEPTED", False):
            fig_s, ax_s = plt.subplots(1, 1, figsize=(10, 4), dpi=150)
            ds = cfg["SEG_PLOT_DOWNSAMPLE"]
            ax_s.plot(t_seg[::ds], I_seg[::ds],        c='tab:blue',   label='Raw')
            ax_s.plot(t_seg[::ds], I_norm_seg[::ds],   c='tab:orange', label='LP baseline')
            ax_s.set_title(f"Segment kept (variation={np.round(seg_variation,4)})")
            ax_s.legend()
            fig_s.tight_layout()
            plt.savefig(os.path.join(
                kept_dir, f"segment_kept_{seg_start}_{seg_end}.png"
            ))
            plt.close(fig_s)

        if (not kept) and cfg.get("SAVE_SEGMENT_DEBUG_WHEN_REJECTED", False):
            fig_s, ax_s = plt.subplots(1, 1, figsize=(10, 4), dpi=150)
            ds = cfg["SEG_PLOT_DOWNSAMPLE"]
            ax_s.plot(t_seg[::ds], I_seg[::ds],        c='tab:blue',   label='Raw')
            ax_s.plot(t_seg[::ds], I_norm_seg[::ds],   c='tab:orange', label='LP baseline')
            ax_s.set_title(f"Segment rejected (variation={np.round(seg_variation,4)})")
            ax_s.legend()
            fig_s.tight_layout()
            plt.savefig(os.path.join(
                disc_dir, f"segment_rejected_{seg_start}_{seg_end}.png"
            ))
            plt.close(fig_s)

        if not kept:
            # skip noisy segments
            continue

        # ---- FEATURE CONSTRUCTION (same logic) ----
        if cfg["FEATURE_MODE"] == "ratio":
            denom_feat = I_norm_seg.copy()
            denom_feat[np.abs(denom_feat) < cfg["EPS_DENOM"]] = cfg["EPS_DENOM"]
            feat = 1.0 - (I_seg / denom_feat)
        else:
            feat = I_seg - I_norm_seg

        if cfg["FEATURE_LP_ENABLE"]:
            if cfg["FEATURE_LP_MODE"] == "frac_fs":
                cutoff = cfg["FEATURE_LP_FRAC"] * fs
            else:
                cutoff = cfg["FEATURE_LP_FIXED_HZ"]
            feat = butter_lowpass_filter(feat, cutoff, fs, cfg["FEATURE_LP_ORDER"])

        feat = feat.astype(float)

        # ---- PEAK DETECTION ----
        min_dist = cfg["PEAK_MIN_DISTANCE"]
        if cfg.get("ENFORCE_DISTANCE_GTE_WINDOW", True):
            min_dist = max(int(min_dist), int(cfg["WINDOW_SIZE"]))

        peak_idxs_local, _ = find_peaks(
            feat,
            prominence=cfg["PEAK_PROMINENCE"],
            distance=min_dist,
        )
        if peak_idxs_local.size == 0:
            continue

        # convert local->global index in the whole trace
        peak_idxs_global = peak_idxs_local + seg_start

        # ---- each peak -> build a window ----
        for local_idx, peak_center in enumerate(peak_idxs_global):

            try:
                # robust half-window in samples
                ws_cfg = int(max(1, cfg["WINDOW_SIZE"]))
                seg_len = seg_end - seg_start
                ws_eff  = min(ws_cfg, max(1, (seg_len - 1)//2))

                # clamp to segment
                start_idx = max(peak_center - ws_eff, seg_start)
                end_idx   = min(peak_center + ws_eff, seg_end)

                local_start = start_idx - seg_start
                local_end   = end_idx   - seg_start
                cur_w       = local_end - local_start

                # enforce minimal width
                min_win = int(max(3, cfg["MIN_WINDOW_SAMPLES"]))
                if cur_w < min_win:
                    need = min_win - cur_w
                    grow_left  = min(need//2 + need%2, local_start)
                    grow_right = min(need//2, seg_len - local_end)
                    local_start -= grow_left
                    local_end   += grow_right
                    start_idx    = seg_start + local_start
                    end_idx      = seg_start + local_end
                    cur_w        = local_end - local_start

                if cur_w < min_win:
                    # still too short, skip
                    continue

                # normalized raw in that window
                denom2 = I_norm[start_idx:end_idx].copy()
                denom2[np.abs(denom2) < cfg["EPS_DENOM"]] = cfg["EPS_DENOM"]
                raw_over_base = (I[start_idx:end_idx] / denom2).astype(float)

                # feature window for this peak
                feature_window = feat[local_start:local_end].astype(float)
                if feature_window.size == 0 or not np.isfinite(feature_window).any():
                    continue

                # build a time axis around peak in µs
                t_slice = (t[start_idx:end_idx] * 1e6) - (t[peak_center] * 1e6)

                # SNR
                if cfg["SNR_REPORT"]:
                    s_ampl  = float(np.max(np.abs(feature_window))) if feature_window.size else 0.0
                    s_noise = float(np.std(feature_window)) if feature_window.size else 1e-18
                    snr_db  = 10.0 * np.log10(
                        (s_ampl / (s_noise if s_noise != 0 else 1e-18)) if s_ampl > 0 else 1e-18
                    )
                else:
                    snr_db = None

                # ---- BUILD PEAK ENTRY (unchanged field names) ----
                peak_idx = global_peak_counter
                peak_dict = {
                    "peak_index":     int(peak_idx),
                    "segment_index":  int(local_idx),
                    "segment_start":  int(seg_start),
                    "segment_end":    int(seg_end),
                    "t_start":        float(t_slice[0]) if t_slice.size else 0.0,
                    "t_end":          float(t_slice[-1]) if t_slice.size else 0.0,
                    "dt":             float(t_slice[1] - t_slice[0]) if t_slice.size > 1 else 0.0,
                    "raw_signal":     raw_over_base.tolist(),
                    "raw_signal_not_norm": I[start_idx:end_idx].astype(float).tolist(),
                    "norm_signal":    I_norm[start_idx:end_idx].astype(float).tolist(),
                    "filtered_signal": feature_window.tolist(),
                    "baseline": float(
                        np.median(I_norm[start_idx:end_idx]) /
                        (np.median(I_norm) if abs(np.median(I_norm)) > 0 else 1.0)
                    ),
                }
                if snr_db is not None:
                    peak_dict["snr_db"] = float(snr_db)

                peak_data.append(peak_dict)

                # ---- per-peak diagnostic plot (same style) ----
                figp, axp = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
                axp.plot(t_slice, feature_window, label='Feature (LP)')
                axp.plot(t_slice, raw_over_base, alpha=0.7, label='Raw / baseline')
                title = f'Peak {peak_idx} (Seg {local_idx})'
                if snr_db is not None:
                    title += f' | SNR: {snr_db:.1f} dB'
                axp.set_title(title)
                axp.set_xlabel('Time (µs)')
                axp.set_ylabel('Amplitude (a.u.)')
                axp.legend()
                figp.tight_layout()
                plt.savefig(os.path.join(figures_dir, f'peak_{peak_idx}.png'))
                plt.close(figp)

                global_peak_counter += 1

            except Exception as e:
                print(f"      Error on peak {global_peak_counter} in {os.path.basename(file_path)}: {e}")
                continue

        # after each kept segment, update JSON on disk (resilient like before)
        with open(os.path.join(output_dir, 'peaks_data.json'), 'w') as f:
            json.dump(peak_data, f, indent=2)

    print(f"  Done: {os.path.basename(file_path)} | total peaks: {len(peak_data)}")


def process_data(root_path: str):
    cfg = CONFIG
    data_folders = get_data_folders(root_path)
    print("Data folders:", data_folders)

    for data_path in data_folders:
        print(f"\nProcessing folder: {data_path}")

        # Collect all files matching ANY of the patterns (["*.dat","*.abf"])
        all_files = []
        for pattern in cfg["FILE_GLOB"]:
            all_files.extend(glob.glob(os.path.join(data_path, pattern)))
        all_files = sorted(all_files)

        if not all_files:
            print("  No matching files.")
            continue

        for file_path in all_files:
            if cfg["SKIP_ALREADY_PROCESSED"] and is_file_processed(root_path, data_path, file_path):
                print(f"  Skipping (already processed): {os.path.basename(file_path)}")
                continue

            process_one_file(file_path, data_path, cfg)

            # mark file as done
            mark_file_as_processed(root_path, data_path, file_path)


def __main__():
    process_data(CONFIG["ROOT_PATH"])

if __name__ == '__main__':
    __main__()
