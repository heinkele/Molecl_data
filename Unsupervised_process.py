from __future__ import annotations
from pathlib import Path
import re
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor, KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from joblib import dump

# -------------------- CONFIG --------------------
SEED = 42
ROOT = Path("/Users/hugo/MOLECL_test/Molecl_data_H")
FEATURE_CSV_NAME = "event_features_with_deltas.csv"
ALL_EVENTS = ROOT / "all_events_labeled.csv"
OUT_DIR = ROOT / "Unsupervised"
PLOTS_DIR = OUT_DIR / "Plots"

# Label columns (propagated if present)
LABEL_COLS = ["sample_label", "sample_folder"]

# Exact feature set you used in the SIM prep
FEATURES_KEEP = [
    "average_blockage", "maximum_blockage",
    "delta_mae_filtered_raw", "delta_mae_fit_raw",
    "kurtosis_excess_raw_resid", "duration_s", "auc_abs",
    "delta_area_filtered_raw", "skewness_raw_resid", "delta_area_fit_raw",
]

# PCA/ICA/GMM parameters
PCA_VAR_TARGET = 0.80
ICA_N_COMPONENTS = 5
GMM_K_MIN, GMM_K_MAX = 2, 10
GMM_ABS_THR = 50.0
GMM_REL_THR = 0.05
GMM_FIXED_K = 8

# Density params
K_NN = 15
K_LOF = 20
KDE_BW_GRID = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0])

# -------------------- HELPERS --------------------
SAMPLE_REGEX = re.compile(r"sample\s*([0-9]+)", re.IGNORECASE)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def infer_sample_label(folder_name: str) -> str:
    m = SAMPLE_REGEX.search(folder_name)
    return f"sample{m.group(1)}" if m else folder_name

def list_sample_csvs(root: Path, csv_name: str = FEATURE_CSV_NAME) -> list[Path]:
    csv_paths: list[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith(".") or "baseline" in p.name.lower():
            continue
        candidate = p / csv_name
        if candidate.exists():
            csv_paths.append(candidate)
    return sorted(csv_paths)

def safe_div(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    out[mask] = a[mask] / b[mask]
    return out

def safe_log(x, eps: float = 1e-12):
    x = np.asarray(x, dtype=float)
    return np.log(np.maximum(x, eps))

def fill_inf_nan(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    for c in cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df

def sample_counts_from_paths(train_csv: Path, test_csv: Path, label_col: str = "sample_label") -> pd.DataFrame:
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    if label_col not in tr.columns or label_col not in te.columns:
        raise ValueError(f"'{label_col}' not found in {train_csv} / {test_csv}")
    cnt_tr = tr[label_col].value_counts().sort_index()
    cnt_te = te[label_col].value_counts().sort_index()
    idx = sorted(set(cnt_tr.index).union(cnt_te.index))
    tbl = pd.DataFrame(index=idx)
    tbl["train"] = cnt_tr.reindex(idx).fillna(0).astype(int)
    tbl["test"]  = cnt_te.reindex(idx).fillna(0).astype(int)
    tbl["total"] = tbl["train"] + tbl["test"]
    return tbl

def write_all_sample_reports(DATASETS: Path, out_dir: Path):
    rep_dir = ensure_dir(out_dir / "Reports")
    long_rows = []
    for folder in sorted(p for p in DATASETS.iterdir() if p.is_dir()):
        train_csv = folder / "train.csv"
        test_csv  = folder / "test.csv"
        if not train_csv.exists() or not test_csv.exists():
            continue
        tbl = sample_counts_from_paths(train_csv, test_csv)
        out_csv = rep_dir / f"{folder.name}_sample_counts.csv"
        tbl.to_csv(out_csv)
        print(f"[Report] {folder.name}\n{tbl.to_string()}\n→ {out_csv}")
        tmp = tbl.reset_index().rename(columns={"index": "sample_label"})
        tmp.insert(0, "dataset", folder.name)
        long_rows.append(tmp)
    if long_rows:
        summary = pd.concat(long_rows, ignore_index=True)
        summary.to_csv(rep_dir / "all_runsets_sample_counts.csv", index=False)


# -------------------- DATA ASSEMBLY --------------------

def build_all_events_labeled() -> pd.DataFrame:
    csv_files = list_sample_csvs(ROOT)
    if not csv_files:
        raise FileNotFoundError(f"No '{FEATURE_CSV_NAME}' files found under {ROOT}")
    frames: list[pd.DataFrame] = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        df["sample_label"] = infer_sample_label(fp.parent.name)
        df["sample_folder"] = fp.parent.name
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    # Reorder columns to show sample info first
    display_cols = LABEL_COLS + [c for c in all_df.columns if c not in LABEL_COLS]
    all_df = all_df[display_cols]
    all_df.to_csv(ALL_EVENTS, index=False)
    print(f"[Assemble] Saved combined CSV to {ALL_EVENTS} with shape {all_df.shape}")
    return all_df

# -------------------- STAGE A: Base (clean + normalize + split) --------------------

def stage_base(all_df: pd.DataFrame) -> dict[str, Path]:
    ensure_dir(OUT_DIR); ensure_dir(PLOTS_DIR)

    # Keep exactly your feature set (drop others if present)
    missing = [c for c in FEATURES_KEEP if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing required features in all_events_labeled.csv: {missing}")
    base = all_df[LABEL_COLS + FEATURES_KEEP].copy()

    # Type enforcement
    for c in FEATURES_KEEP:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    # Log-transform duration (seconds)
    if "duration_s" in base.columns:
        base["duration_s"] = safe_log(base["duration_s"].values)

    # Fill + scale
    num_cols = FEATURES_KEEP.copy()
    base_filled = fill_inf_nan(base, num_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(base_filled[num_cols])
    X_scaled = pd.DataFrame(X_scaled, columns=num_cols, index=base_filled.index)
    selected_normalized = pd.concat([base_filled[LABEL_COLS], X_scaled], axis=1)

    # Save full and split
    BASE_DIR = ensure_dir(OUT_DIR / "Base")
    full_path = BASE_DIR / "base_selected_normalized.csv"
    selected_normalized.to_csv(full_path, index=False)

    # Stratified split by sample_label when available
    if "sample_label" in selected_normalized.columns:
        X = selected_normalized[num_cols].copy()
        y = selected_normalized["sample_label"].copy()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=SEED, stratify=y
        )
        train = pd.concat([y_tr, X_tr], axis=1)
        test  = pd.concat([y_te, X_te], axis=1)
    else:
        idx = np.arange(len(selected_normalized))
        rng = np.random.RandomState(SEED); rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        train = selected_normalized.iloc[idx[:cut]].copy()
        test  = selected_normalized.iloc[idx[cut:]].copy()

    train_path = BASE_DIR / "train.csv"
    test_path  = BASE_DIR / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print("[Stage A] Base done.")
    return {"BASE_DIR": BASE_DIR, "train": train_path, "test": test_path, "full": full_path}

# -------------------- TRACK A: Derived (AB) --------------------

def track_A(all_df: pd.DataFrame, base_paths: dict[str, Path]) -> dict[str, Path]:
    AB_DIR = ensure_dir(OUT_DIR / "A")  # hold A-only normalized

    base = pd.read_csv(base_paths["full"])  # already scaled; we will recompute scaling on augmented

    # Rebuild unscaled numeric from all_df to derive (use original values)
    raw = all_df[LABEL_COLS + FEATURES_KEEP].copy()
    for c in FEATURES_KEEP:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Minimal, safe derived features (no baseline dependency)
    A = pd.DataFrame(index=raw.index)
    # Ratios using blockage levels (avoid division by 0)
    if set(["maximum_blockage", "average_blockage"]).issubset(raw.columns):
        A["ratio_max_to_avg_blockage"] = safe_div(raw["maximum_blockage"], raw["average_blockage"])  # ~shape
    if set(["delta_area_filtered_raw", "delta_area_fit_raw"]).issubset(raw.columns):
        A["ratio_area_filtered_to_fit"] = safe_div(raw["delta_area_filtered_raw"], raw["delta_area_fit_raw"])  # fit vs filtered
    if "auc_abs" in raw.columns and "duration_s" in raw.columns:
        A["auc_per_second"] = safe_div(raw["auc_abs"], raw["duration_s"])  # energy/time

    # Assemble + scale
    full_A = pd.concat([raw[LABEL_COLS], raw[FEATURES_KEEP], A], axis=1)
    num_cols = [c for c in full_A.columns if c not in LABEL_COLS]
    full_A = fill_inf_nan(full_A, num_cols)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(full_A[num_cols].astype(float))
    Xs = pd.DataFrame(Xs, columns=num_cols, index=full_A.index)
    A_norm = pd.concat([full_A[LABEL_COLS], Xs], axis=1)

    A_full_path  = ensure_dir(OUT_DIR / "A") / "trackA_selected_normalized.csv"
    A_train_path = ensure_dir(OUT_DIR / "A") / "trackA_train.csv"
    A_test_path  = ensure_dir(OUT_DIR / "A") / "trackA_test.csv"

    A_norm.to_csv(A_full_path, index=False)

    # Split
    X_A = A_norm[num_cols].copy().astype(float)
    if "sample_label" in A_norm.columns:
        y_A = A_norm["sample_label"].copy()
        XA_tr, XA_te, yA_tr, yA_te = train_test_split(
            X_A, y_A, test_size=0.20, random_state=SEED, stratify=y_A
        )
        train_A = pd.concat([yA_tr, XA_tr], axis=1).reset_index(drop=True)
        test_A  = pd.concat([yA_te, XA_te], axis=1).reset_index(drop=True)
    else:
        idx = np.arange(len(X_A)); rng = np.random.RandomState(SEED); rng.shuffle(idx)
        cut = int(0.8*len(idx))
        train_A = A_norm.iloc[idx[:cut]].reset_index(drop=True)
        test_A  = A_norm.iloc[idx[cut:]].reset_index(drop=True)

    train_A.to_csv(A_train_path, index=False)
    test_A.to_csv(A_test_path, index=False)
    print("[Track A] Derived features done.")
    return {"A_DIR": AB_DIR, "full": A_full_path, "train": A_train_path, "test": A_test_path}

# -------------------- TRACK B: PCA / ICA / Combined --------------------

def track_B(A_paths: dict[str, Path]) -> dict[str, dict[str, Path]]:
    df_full  = pd.read_csv(A_paths["full"])  # includes labels
    df_train = pd.read_csv(A_paths["train"])  # first col is label
    df_test  = pd.read_csv(A_paths["test"])   # first col is label

    feature_cols = [c for c in df_full.columns if c not in LABEL_COLS]
    X_full  = df_full[feature_cols].values
    X_train = df_train[feature_cols].values
    X_test  = df_test[feature_cols].values

    OUT_PCA = ensure_dir(OUT_DIR / "TrackB_PCA"); PLOTS_PCA = ensure_dir(OUT_PCA / "Plots")
    OUT_ICA = ensure_dir(OUT_DIR / "TrackB_ICA"); PLOTS_ICA = ensure_dir(OUT_ICA / "Plots")
    OUT_COMB = ensure_dir(OUT_DIR / "TrackB_Combined")

    # ---- PCA ----
    pca_probe = PCA(random_state=SEED).fit(X_train)
    cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
    k_pca = int(np.argmax(cumvar >= PCA_VAR_TARGET) + 1) if len(cumvar) else 1
    pca = PCA(n_components=k_pca, random_state=SEED).fit(X_train)
    pca_cols = [f"pca_{i+1}" for i in range(k_pca)]

    Z_tr = pca.transform(X_train)
    Z_te = pca.transform(X_test)
    Z_fu = pca.transform(X_full)

    train_pca = pd.concat([df_train.reset_index(drop=True), pd.DataFrame(Z_tr, columns=pca_cols)], axis=1)
    test_pca  = pd.concat([df_test.reset_index(drop=True),  pd.DataFrame(Z_te, columns=pca_cols)], axis=1)
    full_pca  = pd.concat([df_full.reset_index(drop=True),  pd.DataFrame(Z_fu, columns=pca_cols)], axis=1)

    train_pca.to_csv(OUT_PCA / "train_pca.csv", index=False)
    test_pca.to_csv(OUT_PCA / "test_pca.csv", index=False)
    full_pca.to_csv(OUT_PCA / "trackB_pca_full.csv", index=False)

    var_df = pd.DataFrame({"component": pca_cols, "explained_var_ratio": pca.explained_variance_ratio_})
    var_df["cumulative_var"] = var_df["explained_var_ratio"].cumsum()
    var_df.to_csv(OUT_PCA / "pca_variance.csv", index=False)

    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, k_pca+1), var_df["cumulative_var"], marker="o")
    plt.axhline(PCA_VAR_TARGET, linestyle="--", label=f"{int(PCA_VAR_TARGET*100)}%")
    plt.ylim(0,1.02); plt.xlabel("# components"); plt.ylabel("Cum. explained var")
    plt.title("PCA cumulative variance"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_PCA / "Plots" / "pca_cumulative_variance.png"); plt.close()

    loadings = pd.DataFrame(pca.components_.T, index=feature_cols, columns=pca_cols)
    loadings.to_csv(OUT_PCA / "pca_loadings.csv")

    # ---- ICA ----
    n_ica = min(ICA_N_COMPONENTS, X_train.shape[1]) if X_train.size else 1
    ica = FastICA(n_components=n_ica, random_state=SEED, max_iter=1000, tol=0.001).fit(X_train)
    ica_cols = [f"ica_{i+1}" for i in range(ica.components_.shape[0])]

    S_tr = ica.transform(X_train)
    S_te = ica.transform(X_test)
    S_fu = ica.transform(X_full)

    train_ica = pd.concat([df_train.reset_index(drop=True), pd.DataFrame(S_tr, columns=ica_cols)], axis=1)
    test_ica  = pd.concat([df_test.reset_index(drop=True),  pd.DataFrame(S_te, columns=ica_cols)], axis=1)
    full_ica  = pd.concat([df_full.reset_index(drop=True),  pd.DataFrame(S_fu, columns=ica_cols)], axis=1)

    train_ica.to_csv(OUT_ICA / "train_ica.csv", index=False)
    test_ica.to_csv(OUT_ICA / "test_ica.csv", index=False)
    full_ica.to_csv(OUT_ICA / "trackB_ica_full.csv", index=False)

    mixing = pd.DataFrame(ica.mixing_, index=feature_cols, columns=ica_cols)
    mixing.to_csv(OUT_ICA / "ica_mixing_matrix.csv")

    # ---- Combined ----
    train_comb = pd.concat([df_train.reset_index(drop=True), pd.DataFrame(Z_tr, columns=pca_cols), pd.DataFrame(S_tr, columns=ica_cols)], axis=1)
    test_comb  = pd.concat([df_test.reset_index(drop=True),  pd.DataFrame(Z_te, columns=pca_cols), pd.DataFrame(S_te, columns=ica_cols)], axis=1)
    full_comb  = pd.concat([df_full.reset_index(drop=True),  pd.DataFrame(Z_fu, columns=pca_cols), pd.DataFrame(S_fu, columns=ica_cols)], axis=1)

    ensure_dir(OUT_COMB)
    train_comb.to_csv(OUT_COMB / "train_pca_ica.csv", index=False)
    test_comb.to_csv(OUT_COMB / "test_pca_ica.csv", index=False)
    full_comb.to_csv(OUT_COMB / "trackB_full_pca_ica.csv", index=False)

    return {
        "PCA": {"OUT": OUT_PCA, "train": OUT_PCA/"train_pca.csv", "test": OUT_PCA/"test_pca.csv", "full": OUT_PCA/"trackB_pca_full.csv", "pca_cols": pca_cols},
        "ICA": {"OUT": OUT_ICA, "train": OUT_ICA/"train_ica.csv", "test": OUT_ICA/"test_ica.csv", "full": OUT_ICA/"trackB_ica_full.csv"},
        "COMB": {"OUT": OUT_COMB, "train": OUT_COMB/"train_pca_ica.csv", "test": OUT_COMB/"test_pca_ica.csv", "full": OUT_COMB/"trackB_full_pca_ica.csv", "pca_cols": pca_cols, "ica_cols": ica_cols}
    }

# -------------------- TRACK C: GMM on PCA --------------------

def track_C_gmm(pca_paths: dict[str, Path]):
    df_train = pd.read_csv(pca_paths["train"]) ; df_test = pd.read_csv(pca_paths["test"]) ; df_full = pd.read_csv(pca_paths["full"])
    pca_cols = [c for c in df_full.columns if c.startswith("pca_")]
    X_tr, X_te, X_fu = df_train[pca_cols].values, df_test[pca_cols].values, df_full[pca_cols].values

    OUT_GMM = ensure_dir(OUT_DIR / "TrackC_GMM"); PLOTS = ensure_dir(OUT_GMM / "Plots")

    # Model selection
    bic_list, aic_list, models = [], [], []
    for k in range(GMM_K_MIN, GMM_K_MAX+1):
        g = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED, n_init=5, reg_covar=1e-6, init_params="kmeans").fit(X_tr)
        models.append(g)
        bic_list.append(g.bic(X_tr)); aic_list.append(g.aic(X_tr))

    k_values = np.arange(GMM_K_MIN, GMM_K_MAX+1)
    bic_arr = np.array(bic_list); d_bic = np.diff(bic_arr)
    chosen_idx = None
    for i in range(1, len(k_values)):
        abs_impr = -d_bic[i-1]
        rel_impr = abs_impr / (abs(bic_arr[i-1]) + 1e-12)
        if (abs_impr < GMM_ABS_THR) or (rel_impr < GMM_REL_THR):
            chosen_idx = i; break
    if chosen_idx is None:
        chosen_idx = int(np.argmin(bic_arr))
    best_k = int(k_values[chosen_idx]); best_gmm = models[chosen_idx]

    sel_df = pd.DataFrame({"K": k_values, "BIC": bic_list, "AIC": aic_list, "dBIC_from_prev": np.r_[np.nan, d_bic]})
    sel_df.to_csv(OUT_GMM / "gmm_model_selection.csv", index=False)

    plt.figure(figsize=(7,4))
    plt.plot(k_values, bic_list, marker="o", label="BIC")
    plt.plot(k_values, aic_list, marker="o", alpha=0.7, label="AIC")
    plt.axvline(best_k, color="k", linestyle="--", label=f"Chosen K={best_k}")
    k_argmin = int(k_values[np.argmin(bic_arr)])
    plt.axvline(k_argmin, color="gray", linestyle=":", label=f"Argmin BIC K={k_argmin}")
    plt.xlabel("K"); plt.ylabel("Criterion (↓)"); plt.title("GMM model selection"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "gmm_bic_aic.png"); plt.close()

    # Fit fixed-K model too (K=8)
    gmm_fixed = GaussianMixture(n_components=GMM_FIXED_K, covariance_type="full", random_state=SEED, n_init=5, reg_covar=1e-6, init_params="kmeans").fit(X_tr)

    def augment(df_in: pd.DataFrame, X: np.ndarray, gmm: GaussianMixture, tag: str) -> pd.DataFrame:
        prob = gmm.predict_proba(X)
        lbls = gmm.predict(X)
        logp = gmm.score_samples(X)
        out = df_in.reset_index(drop=True).copy()
        prob_df = pd.DataFrame(prob, columns=[f"{tag}_p{k+1}" for k in range(gmm.n_components)])
        out = pd.concat([out, prob_df], axis=1)
        out[f"{tag}_cluster"] = lbls
        out[f"{tag}_logpdf"]  = logp
        return out

    # Augment with chosen-K and fixed-K
    train_best = augment(df_train, X_tr, best_gmm,  f"gmmK{best_k}")
    test_best  = augment(df_test,  X_te, best_gmm,  f"gmmK{best_k}")
    full_best  = augment(df_full,  X_fu, best_gmm,  f"gmmK{best_k}")

    train_fixed = augment(df_train, X_tr, gmm_fixed, f"gmmK{GMM_FIXED_K}")
    test_fixed  = augment(df_test,  X_te, gmm_fixed, f"gmmK{GMM_FIXED_K}")
    full_fixed  = augment(df_full,  X_fu, gmm_fixed, f"gmmK{GMM_FIXED_K}")

    train_best.to_csv(OUT_GMM / f"trackABC_train_gmmK{best_k}.csv", index=False)
    test_best.to_csv(OUT_GMM / f"trackABC_test_gmmK{best_k}.csv", index=False)
    full_best.to_csv(OUT_GMM / f"trackABC_full_gmmK{best_k}.csv", index=False)

    train_fixed.to_csv(OUT_GMM / f"trackABC_train_gmmK{GMM_FIXED_K}.csv", index=False)
    test_fixed.to_csv(OUT_GMM / f"trackABC_test_gmmK{GMM_FIXED_K}.csv", index=False)
    full_fixed.to_csv(OUT_GMM / f"trackABC_full_gmmK{GMM_FIXED_K}.csv", index=False)

    dump(best_gmm, OUT_GMM / f"gmmK{best_k}_model.joblib")
    dump(gmm_fixed, OUT_GMM / f"gmmK{GMM_FIXED_K}_model.joblib")
    print(f"[Track C] Done. Best K={best_k}, also fitted fixed K={GMM_FIXED_K}.")

    return {"OUT": OUT_GMM, "best_k": best_k, "pca_cols": pca_cols}

# -------------------- TRACK D: Density on PCA --------------------

def track_D_density(pca_paths: dict[str, Path]):
    df_train = pd.read_csv(pca_paths["train"]) ; df_test = pd.read_csv(pca_paths["test"]) ; df_full = pd.read_csv(pca_paths["full"])
    pca_cols = [c for c in df_full.columns if c.startswith("pca_")]
    X_tr, X_te, X_fu = df_train[pca_cols].values, df_test[pca_cols].values, df_full[pca_cols].values

    OUT_DEN = ensure_dir(OUT_DIR / "TrackD_Density"); PLOTS = ensure_dir(OUT_DEN / "Plots")

    # kNN distances
    nn = NearestNeighbors(n_neighbors=K_NN, algorithm="auto", metric="euclidean")
    nn.fit(X_tr)

    def knn_stats(Xq):
        dists, _ = nn.kneighbors(Xq, n_neighbors=K_NN, return_distance=True)
        return dists.mean(axis=1), dists[:, -1]

    tr_mean, tr_kth = knn_stats(X_tr)
    te_mean, te_kth = knn_stats(X_te)
    fu_mean, fu_kth = knn_stats(X_fu)

    D = X_tr.shape[1]; EPS = 1e-12
    def pseudo_density(kth_radius):
        return K_NN / (np.power(kth_radius + EPS, D))

    tr_rho, te_rho, fu_rho = pseudo_density(tr_kth), pseudo_density(te_kth), pseudo_density(fu_kth)

    # LOF novelty (fit on train, score everywhere)
    lof = LocalOutlierFactor(n_neighbors=K_LOF, novelty=True)
    lof.fit(X_tr)
    lof_tr = -lof.score_samples(X_tr)  # higher = more outlier‑y
    lof_te = -lof.score_samples(X_te)
    lof_fu = -lof.score_samples(X_fu)

    # KDE with CV bandwidth on train
    grid = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": KDE_BW_GRID}, cv=5, n_jobs=-1)
    grid.fit(X_tr)
    kde_best = grid.best_estimator_; best_bw = kde_best.bandwidth
    log_kde_tr = kde_best.score_samples(X_tr)
    log_kde_te = kde_best.score_samples(X_te)
    log_kde_fu = kde_best.score_samples(X_fu)

    def attach(df_base, mean_d, kth_d, rho, lof_s, log_kde):
        out = df_base.reset_index(drop=True).copy()
        out[f"knn_dist_mean_{K_NN}"] = mean_d
        out[f"knn_dist_k_{K_NN}"] = kth_d
        out[f"knn_density_k_{K_NN}"] = rho
        out[f"lof_{K_LOF}"] = lof_s
        out[f"log_kde_bw{best_bw:.2f}"] = log_kde
        return out

    train_den = attach(df_train, tr_mean, tr_kth, tr_rho, lof_tr, log_kde_tr)
    test_den  = attach(df_test,  te_mean, te_kth, te_rho, lof_te, log_kde_te)
    full_den  = attach(df_full,  fu_mean, fu_kth, fu_rho, lof_fu, log_kde_fu)

    train_out = OUT_DEN / "trackD_train_density.csv"
    test_out  = OUT_DEN / "trackD_test_density.csv"
    full_out  = OUT_DEN / "trackD_full_density.csv"
    train_den.to_csv(train_out, index=False)
    test_den.to_csv(test_out, index=False)
    full_den.to_csv(full_out, index=False)

    dump(nn, OUT_DEN / f"nn_k{K_NN}.joblib")
    dump(lof, OUT_DEN / f"lof_k{K_LOF}_novel.joblib")
    dump(kde_best, OUT_DEN / f"kde_bandwidth_{best_bw:.2f}.joblib")

    print(f"[Track D] Density done. Best KDE bw={best_bw:.2f}")
    return {"OUT": OUT_DEN, "pca_cols": pca_cols}

# -------------------- DATASET BUNDLES for SVM --------------------

def make_runsets_for_svm(base_paths, A_paths, B_paths, C_paths, D_paths):
    DATASETS = ensure_dir(OUT_DIR / "Datasets")

    # Base
    df_tr_base = pd.read_csv(base_paths["train"]) ; df_te_base = pd.read_csv(base_paths["test"]) ;
    ensure_dir(DATASETS/"Base")
    df_tr_base.to_csv(DATASETS/"Base"/"train.csv", index=False)
    df_te_base.to_csv(DATASETS/"Base"/"test.csv", index=False)

    # A only
    df_tr_A = pd.read_csv(A_paths["train"]) ; df_te_A = pd.read_csv(A_paths["test"]) ;
    ensure_dir(DATASETS/"A")
    df_tr_A.to_csv(DATASETS/"A"/"train.csv", index=False)
    df_te_A.to_csv(DATASETS/"A"/"test.csv", index=False)

    # AB (PCA only appended to A train/test) – here we simply reuse PCA outputs
    df_tr_p = pd.read_csv(B_paths["PCA"]["train"]) ; df_te_p = pd.read_csv(B_paths["PCA"]["test"]) ;
    ensure_dir(DATASETS/"AB")
    df_tr_p.to_csv(DATASETS/"AB"/"train.csv", index=False)
    df_te_p.to_csv(DATASETS/"AB"/"test.csv", index=False)

    # ABC (GMM fixed‑K columns merged to PCA) – already contains labels+pca+gmm
    df_tr_gmm = pd.read_csv(OUT_DIR/"TrackC_GMM"/f"trackABC_train_gmmK{GMM_FIXED_K}.csv")
    df_te_gmm = pd.read_csv(OUT_DIR/"TrackC_GMM"/f"trackABC_test_gmmK{GMM_FIXED_K}.csv")
    ensure_dir(DATASETS/"ABC")
    df_tr_gmm.to_csv(DATASETS/"ABC"/"train.csv", index=False)
    df_te_gmm.to_csv(DATASETS/"ABC"/"test.csv", index=False)

    # ABD (Density features joined to PCA)
    df_tr_den = pd.read_csv(D_paths["OUT"]/"trackD_train_density.csv")
    df_te_den = pd.read_csv(D_paths["OUT"]/"trackD_test_density.csv")
    ensure_dir(DATASETS/"ABD")
    df_tr_den.to_csv(DATASETS/"ABD"/"train.csv", index=False)
    df_te_den.to_csv(DATASETS/"ABD"/"test.csv", index=False)

    print("[Runsets] Prepared Base, A, AB, ABC, ABD datasets for SVM.")
    return DATASETS

# -------------------- TRACK E: SVM --------------------

def run_svm(DATASETS: Path):
    OUT = ensure_dir(OUT_DIR / "SVM_Results"); PLOTS = ensure_dir(OUT / "Plots"); MODELS = ensure_dir(OUT / "Models")

    def find_csv(folder: Path, kw: str) -> Path:
        cands = [f for f in os.listdir(folder) if f.lower().endswith(".csv") and kw.lower() in f.lower()]
        if not cands: raise FileNotFoundError(f"No *{kw}*.csv found in {folder}")
        cands.sort(key=lambda x: (len(x), x.lower()))
        return folder / cands[0]

    def load_xy(path: Path):
        df = pd.read_csv(path)
        if "sample_label" not in df.columns:
            raise ValueError(f"'sample_label' column not found in {path}")
        X_cols = [c for c in df.columns if c not in LABEL_COLS]
        X = df[X_cols].values
        y = df["sample_label"].values
        return df, X, y, X_cols

    RUNSETS = ["Base", "A", "AB", "ABC", "ABD"]

    param_grid = {
        "svc__C":    [0.5, 1, 2, 5, 10],
        "svc__gamma": ["scale", 0.01, 0.05, 0.1, 0.2],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    summary_rows = []

    for ds in RUNSETS:
        folder = DATASETS / ds
        if not folder.is_dir():
            print(f"Skipping {ds}: folder not found -> {folder}")
            continue

        print(f"\n=== Dataset: {ds} ===")
        train_csv = find_csv(folder, "train")
        test_csv  = find_csv(folder, "test")

        df_train, X_train, y_train, feat_cols = load_xy(train_csv)
        df_test,  X_test,  y_test,  _        = load_xy(test_csv)

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svc", SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=SEED)),
        ])

        gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=0, refit=True)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_; best_params = gs.best_params_
        print(f"Best params ({ds}): {best_params} | CV macro-F1={gs.best_score_:.3f}")

        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bal = balanced_accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        f1w = f1_score(y_test, y_pred, average="weighted")

                # Normalized confusion matrix (row-normalized -> per-class confidence 0..1)
        labels = np.unique(np.concatenate([y_train, y_test]))
        cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")

        fig = plt.figure(figsize=(5.8, 4.6))
        im = plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        plt.title(f"{ds} – Confusion Matrix (row-normalized) • Macro-F1={f1m:.3f}")
        plt.xlabel("Predicted"); plt.ylabel("True")

        plt.xticks(range(len(labels)), labels, rotation=0)
        plt.yticks(range(len(labels)), labels)

        # Annotate cells with values in [0,1]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                plt.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=9,
                    color=("white" if val > 0.5 else "black")
                )

        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Per-class confidence (row sum = 1)")

        plt.tight_layout()
        cm_path = PLOTS / f"{ds}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close(fig)

        report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
        with open(OUT / f"{ds}_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        dump(best, MODELS / f"svm_{ds}.joblib")

        summary_rows.append({
            "dataset": ds,
            "n_features": len(feat_cols),
            "best_C": best_params["svc__C"],
            "best_gamma": best_params["svc__gamma"],
            "test_accuracy": acc,
            "test_balanced_acc": bal,
            "test_f1_macro": f1m,
            "test_f1_weighted": f1w,
            "confusion_matrix_png": str(cm_path),
        })

        # Permutation importance
        perm = permutation_importance(best, X_test, y_test, scoring="f1_macro", n_repeats=20, random_state=SEED, n_jobs=-1)
        imp_df = pd.DataFrame({
            "feature": feat_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False)
        imp_csv = OUT / f"{ds}_permutation_importance.csv"
        imp_df.to_csv(imp_csv, index=False)

        topN = min(10, len(imp_df))
        top_df = imp_df.head(topN)[::-1]
        plt.figure(figsize=(8,5))
        plt.barh(top_df["feature"], top_df["importance_mean"], xerr=top_df["importance_std"], alpha=0.9)
        plt.xlabel("Permutation importance (Δ macro‑F1)")
        plt.title(f"{ds} – Top {topN} feature importances (RBF SVM)")
        plt.tight_layout()
        fi_path = PLOTS / f"{ds}_feature_importance_top{topN}.png"
        plt.savefig(fi_path); plt.close()
        print(f"[{ds}] Saved permutation importance: {imp_csv} and {fi_path}")

    summary_df = pd.DataFrame(summary_rows).sort_values(by="test_f1_macro", ascending=False)
    summary_path = OUT / "svm_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n[Track E] SVM done.")
    print(f"Summary: {summary_path}\nPlots: {PLOTS}\nModels: {MODELS}")

    # -------------------- MAIN --------------------

def __main__():
    ensure_dir(OUT_DIR)
    if not ALL_EVENTS.exists():
        all_df = build_all_events_labeled()
    else:
        all_df = pd.read_csv(ALL_EVENTS)
        print(f"[Assemble] Using existing {ALL_EVENTS} with shape {all_df.shape}")

    # Stage A (Base)
    base_paths = stage_base(all_df)

    # Track A (Derived)
    A_paths = track_A(all_df, base_paths)

    # Track B (PCA/ICA/Combined)
    B_paths = track_B(A_paths)

    # Track C (GMM on PCA)
    C_paths = track_C_gmm(B_paths["PCA"])

    # Track D (Density on PCA)
    D_paths = track_D_density(B_paths["PCA"])

    # Bundle datasets for SVM
    DATASETS = make_runsets_for_svm(base_paths, A_paths, B_paths, C_paths, D_paths)

    # Track E (SVM)
    run_svm(DATASETS)

    # Sample-count reports for each runset (Base, A, AB, ABC, ABD)
    write_all_sample_reports(DATASETS, OUT_DIR)

if __name__ == '__main__':
    __main__()