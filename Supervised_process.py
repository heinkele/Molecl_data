from __future__ import annotations
from pathlib import Path
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.inspection import permutation_importance

# ------------------- CONFIG -------------------
SEED = 42
ROOT = Path("/Users/hugo/New data/PacBio")
FEATURE_CSV_NAME = "event_features_with_deltas.csv"  # per-sample file
ALL_EVENTS = ROOT / "all_events_labeled.csv"         # concatenated file
OUT_DIR = ROOT / "Supervised" / "ModelSelection"
PLOTS_DIR = OUT_DIR / "Plots"

# Labels present in Molecl combined CSV
LABEL_COLS = ["sample_label", "sample_folder"]
OPTIONAL_LABELS = ["Condition"]  # only if exists

# SIM / SVM feature set (must exist in ALL_EVENTS)
FEATURES_KEEP = [
    "average_blockage", "maximum_blockage",
    "delta_mae_filtered_raw", "delta_mae_fit_raw",
    "kurtosis_excess_raw_resid", "duration_s", "auc_abs",
    "delta_area_filtered_raw", "skewness_raw_resid", "delta_area_fit_raw",
    "noise_delta_area_filtered_raw",
    "noise_delta_mae_filtered_raw",
    "noise_delta_area_fit_raw",
    "noise_delta_mae_fit_raw",
]

# ------------------- HELPERS -------------------
SAMPLE_REGEX = re.compile(r"sample\s*([0-9]+)", re.IGNORECASE)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def infer_sample_label(folder_name: str) -> str:
    m = SAMPLE_REGEX.search(folder_name)
    return f"sample{m.group(1)}" if m else folder_name

def list_sample_csvs(root: Path, csv_name: str = FEATURE_CSV_NAME) -> list[Path]:
    csvs: list[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith(".") or "baseline" in p.name.lower():
            continue
        cand = p / csv_name
        if cand.exists():
            csvs.append(cand)
    return sorted(csvs)

def write_split_sample_report(y_train: pd.Series, y_test: pd.Series, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    tr = y_train.value_counts().sort_index()
    te = y_test.value_counts().sort_index()
    idx = sorted(set(tr.index).union(te.index))
    tbl = pd.DataFrame({
        "train": tr.reindex(idx).fillna(0).astype(int),
        "test":  te.reindex(idx).fillna(0).astype(int)
    })
    tbl["total"] = tbl["train"] + tbl["test"]
    csv_path = out_dir / "sample_counts_split.csv"
    tbl.to_csv(csv_path)
    print(f"[Report] Train/Test sample counts\n{tbl.to_string()}\n→ {csv_path}")


# ------------------- DATA ASSEMBLY -------------------

def build_all_events_labeled() -> pd.DataFrame:
    files = list_sample_csvs(ROOT)
    if not files:
        raise FileNotFoundError(f"No '{FEATURE_CSV_NAME}' under {ROOT}")
    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        df["sample_label"] = infer_sample_label(fp.parent.name)
        df["sample_folder"] = fp.parent.name
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    cols = LABEL_COLS + [c for c in all_df.columns if c not in LABEL_COLS]
    all_df = all_df[cols]
    all_df.to_csv(ALL_EVENTS, index=False)
    print(f"[Assemble] Wrote {ALL_EVENTS} shape={all_df.shape}")
    return all_df

# ------------------- NORMALIZATION -------------------

def normalize_per_batch(all_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required features exist
    missing = [c for c in FEATURES_KEEP if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing features in all_events_labeled.csv: {missing}")

    # Keep only labels + features
    cols = LABEL_COLS + [c for c in OPTIONAL_LABELS if c in all_df.columns] + FEATURES_KEEP
    df = all_df[cols].copy()

    # Numeric coercion
    for c in FEATURES_KEEP:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Log-transform duration
    if "duration_s" in df.columns:
        df["duration_s"] = np.log(np.maximum(df["duration_s"].astype(float), 1e-12))

    # Batch keys: by sample_label (+ Condition if present)
    by = ["sample_label"] + (["Condition"] if "Condition" in df.columns else [])

    def zscore_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for f in FEATURES_KEEP:
            x = g[f].astype(float)
            mu = x.mean()
            sd = x.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                g[f + "_z"] = 0.0  # constant group => zero variance safeguard
            else:
                g[f + "_z"] = (x - mu) / sd
        return g

    df_norm = df.groupby(by, group_keys=False).apply(zscore_group)

    # Final table: keep labels + z-scored features
    zcols = [f + "_z" for f in FEATURES_KEEP]
    out = pd.concat([df[LABEL_COLS + [c for c in OPTIONAL_LABELS if c in df.columns]].reset_index(drop=True),
                     df_norm[zcols].reset_index(drop=True)], axis=1)
    return out

# ------------------- GRID SEARCH PIPELINES -------------------

def run_gridsearch(X_train: pd.DataFrame, y_train: pd.Series, out_csv: Path):
    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("reduce", PCA()),  # placeholder; overridden by param_grid
        ("clf", SVC(class_weight="balanced", probability=False, random_state=SEED)),
    ])

    param_grid = [
        {   # PCA pipelines
            "reduce":               [PCA()],
            "reduce__n_components": [3, 4, 5, 6, 7],
            "clf__kernel":          ["rbf"],
            "clf__C":               [1, 5],
            "clf__gamma":           ["scale", "auto"],
        },
        {   # LDA pipelines (n_components ≤ n_classes-1)
            "reduce":               [LDA()],
            "reduce__n_components": [1, 2],
            "clf__kernel":          ["linear"],
            "clf__C":               [0.1, 1],
        },
        {   # SelectKBest pipelines
            "reduce":               [SelectKBest(score_func=f_classif)],
            "reduce__k":            [5, 7, 9, 11],
            "clf__kernel":          ["rbf"],
            "clf__C":               [1, 5],
            "clf__gamma":           ["scale", "auto"],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
        return_train_score=False,
    )
    grid.fit(X_train, y_train)

    results = pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False)
    results.to_csv(out_csv, index=False)
    print(f"[Grid] Saved results → {out_csv}")
    return grid, results

# ------------------- ANALYSIS & PLOTS -------------------

def label_reducer(val) -> str:
    s = str(val)
    if "PCA" in s: return "PCA"
    if "SelectKBest" in s: return "SelectKBest"
    if "LDA" in s or "LinearDiscriminantAnalysis" in s: return "LDA"
    return s

def plots_from_grid(results: pd.DataFrame, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    df = results.copy()
    # n_feat column
    df["reducer"] = df["param_reduce"].astype(str).apply(label_reducer)
    df["n_feat"] = (
        df["param_reduce__n_components"].fillna(df["param_reduce__k"]).astype(float)
    )

    # Heatmaps for PCA/SelectKBest (RBF only)
    for method in ["PCA", "SelectKBest"]:
        sub = df[(df["reducer"] == method) & (df["param_clf__kernel"] == "rbf")].copy()
        if sub.empty: continue
        # pick the n with best mean score
        best_n = sub.groupby("n_feat")["mean_test_score"].mean().idxmax()
        heat = sub[sub["n_feat"] == best_n]
        if heat.empty: continue
        pivot = heat.pivot(index="param_clf__C", columns="param_clf__gamma", values="mean_test_score")
        plt.figure(figsize=(4,3))
        plt.imshow(pivot, origin="lower", aspect="auto", interpolation="nearest")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45)
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        for (i,j), val in np.ndenumerate(pivot.values):
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)
        plt.colorbar(label="Mean CV Macro‑F1")
        plt.title(f"{method} (n={int(best_n)}) • RBF")
        plt.xlabel("γ"); plt.ylabel("C"); plt.tight_layout()
        plt.savefig(out_dir / f"heatmap_{method.lower()}_rbf.png"); plt.close()

    # LDA heatmap (linear)
    sub_lda = df[(df["reducer"] == "LDA") & (df["param_clf__kernel"] == "linear")]
    if not sub_lda.empty:
        pivot_lda = sub_lda.pivot(index="param_clf__C", columns="n_feat", values="mean_test_score")
        plt.figure(figsize=(4,3))
        plt.imshow(pivot_lda, origin="lower", aspect="auto", interpolation="nearest")
        plt.xticks(np.arange(len(pivot_lda.columns)), pivot_lda.columns)
        plt.yticks(np.arange(len(pivot_lda.index)), pivot_lda.index)
        for (i,j), val in np.ndenumerate(pivot_lda.values):
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)
        plt.colorbar(label="Mean CV Macro‑F1")
        plt.title("LDA • Linear"); plt.xlabel("n_components"); plt.ylabel("C")
        plt.tight_layout(); plt.savefig(out_dir / "heatmap_lda_linear.png"); plt.close()

    # Performance vs complexity scatter
    plt.figure(figsize=(6,4))
    plt.scatter(df["n_feat"], df["mean_test_score"], alpha=0.35, label="all models")
    plt.xlabel("# Components / Features"); plt.ylabel("Mean CV Macro‑F1")
    plt.title("Performance vs Complexity"); plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(out_dir / "perf_vs_complexity.png"); plt.close()

def plot_perm_importance(model, X, y, feat_names, title, outfile, seed=42):
    """Permutation importance on the full pipeline (pre-reduction features)."""
    res = permutation_importance(
        model, X, y,
        scoring="f1_macro",
        n_repeats=20,
        random_state=seed,
        n_jobs=-1
    )
    imp_df = (
        pd.DataFrame({
            "feature": feat_names,
            "mean": res.importances_mean,
            "std":  res.importances_std
        })
        .sort_values("mean", ascending=False)
        .head(10)[::-1]  # reverse for nice bottom-to-top bars
    )

    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["feature"], imp_df["mean"], xerr=imp_df["std"], alpha=0.9)
    plt.xlabel("Permutation importance (Δ macro-F1)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"[PermImportance] Saved → {outfile}")


# ------------------- ONE‑STD‑ERROR RULE -------------------

def one_std_error_choice(results: pd.DataFrame) -> pd.Series:
    df = results.copy()
    df["n_feat"] = df["param_reduce__n_components"].fillna(df["param_reduce__k"]).astype(float)
    best_idx = df["mean_test_score"].idxmax()
    mu_star = df.loc[best_idx, "mean_test_score"]
    sigma_star = df.loc[best_idx, "std_test_score"]
    thr = mu_star - sigma_star
    cand = df[df["mean_test_score"] >= thr]
    opt_idx = cand["n_feat"].idxmin()
    return df.loc[opt_idx]

# ------------------- MAIN -------------------

def __main__():
    ensure_dir(OUT_DIR); ensure_dir(PLOTS_DIR)

    # 1) Build or load all_events
    if ALL_EVENTS.exists():
        all_df = pd.read_csv(ALL_EVENTS)
        print(f"[Assemble] Using existing {ALL_EVENTS} shape={all_df.shape}")
    else:
        all_df = build_all_events_labeled()

    # 2) Normalize per batch and save
    df_norm = normalize_per_batch(all_df)
    norm_path = OUT_DIR / "events_normalized.csv"
    df_norm.to_csv(norm_path, index=False)
    print(f"[Normalize] Saved → {norm_path}")

    # 3) Train/test split (stratified)
    X = df_norm.drop(columns=[c for c in df_norm.columns if c in LABEL_COLS or c in OPTIONAL_LABELS])
    y = df_norm["sample_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    (OUT_DIR / "Splits").mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUT_DIR/"Splits"/"X_train.csv", index=False)
    X_test.to_csv(OUT_DIR/"Splits"/"X_test.csv", index=False)
    y_train.to_csv(OUT_DIR/"Splits"/"y_train.csv", index=False)
    y_test.to_csv(OUT_DIR/"Splits"/"y_test.csv", index=False)
    print(f"[Split] Train={X_train.shape[0]} Test={X_test.shape[0]}")
    # short report of counts per sample in train/test
    write_split_sample_report(y_train, y_test, OUT_DIR / "Reports")

    # 4) Grid search
    grid_csv = OUT_DIR / "grid_results.csv"
    grid, results = run_gridsearch(X_train, y_train, grid_csv)

    # 5) Plots
    plots_from_grid(results, PLOTS_DIR)

    # 6) One‑Std‑Error selection
    opt = one_std_error_choice(results)
    print("\n[1‑SE] Choice:")
    print(opt[[
        "param_reduce", "param_reduce__n_components", "param_reduce__k",
        "param_clf__kernel", "param_clf__C", "param_clf__gamma",
        "mean_test_score", "std_test_score"
    ]])

    # 7) Rebuild chosen pipeline and fit on train
    # Reducer
    reducer_str = str(opt["param_reduce"])
    if "PCA" in reducer_str:
        reducer = PCA(n_components=int(opt["param_reduce__n_components"]))
    elif "LinearDiscriminantAnalysis" in reducer_str or "LDA" in reducer_str:
        reducer = LDA(n_components=int(opt["param_reduce__n_components"]))
    else:
        reducer = SelectKBest(score_func=f_classif, k=int(opt["param_reduce__k"]))

    # Classifier
    svc_params = {
        "kernel": opt["param_clf__kernel"],
        "C": opt["param_clf__C"],
        "class_weight": "balanced",
        "random_state": SEED,
    }
    if svc_params["kernel"] in ["rbf", "poly"] and pd.notna(opt.get("param_clf__gamma", np.nan)):
        svc_params["gamma"] = opt["param_clf__gamma"]
    clf = SVC(**svc_params)

    best_pipe = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("reduce", reducer),
        ("clf", clf),
    ])
    best_pipe.fit(X_train, y_train)

    # 8) Evaluate on test
    y_pred = best_pipe.predict(X_test)

    # Feature importance plot
    plot_title = "A – Top 10 feature importances (RBF SVM)"
    plot_path  = PLOTS_DIR / "A_feature_importance_top10.png"
    plot_perm_importance(best_pipe, X_test, y_test, X_train.columns, plot_title, plot_path, seed=SEED)

    print("\n[Test] Classification report:\n", classification_report(y_test, y_pred, digits=3))
    bal = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"Balanced acc: {bal:.3f} | Macro-F1: {f1m:.3f}")

    # Normalized confusion matrix (per-row → confidence 0..1)
    labels = np.unique(y_train)
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")

    plt.figure(figsize=(5,4))
    im = plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.title(f"Confusion Matrix (Test) • Macro-F1={f1m:.3f}")
    plt.xlabel("Predicted"); plt.ylabel("True")

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)

    # Annotate each cell with value in [0,1]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                     color=("white" if cm[i, j] > 0.5 else "black"))

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Per-class confidence (row-normalized)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix_test.png")
    plt.close()

    # 9) CV stability of chosen model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(best_pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"[CV] Macro‑F1 over folds: mean={cv_scores.mean():.3f} std={cv_scores.std():.3f}")
    plt.figure(figsize=(4,3))
    plt.boxplot(cv_scores, showmeans=True)
    plt.title("CV Macro‑F1 (Chosen Model)"); plt.ylabel("Macro‑F1")
    plt.tight_layout(); plt.savefig(PLOTS_DIR/"cv_macroF1_boxplot.png"); plt.close()

    # 10) If SelectKBest chosen, list final features
    if isinstance(reducer, SelectKBest):
        mask = reducer.get_support()
        selected = X_train.columns[mask]
        with open(OUT_DIR/"selected_features.txt", "w") as f:
            f.write("\n".join(selected))
        print("[SelectKBest] Final features → selected_features.txt")

if __name__ == '__main__':
    __main__()