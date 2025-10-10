from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path("/Users/hugo/MOLECL_test/Molecl_data_H")

# If your sample folder names always contain "...sample<digit>...", this regex will extract it.
SAMPLE_REGEX = re.compile(r"sample\s*([0-9]+)", re.IGNORECASE)

# The per-sample CSV produced by your pipeline
FEATURE_CSV_NAME = "event_features_with_deltas.csv"

def list_sample_csvs(root: Path, csv_name: str = FEATURE_CSV_NAME):
    csv_paths = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        # skip hidden and non-sample dirs
        if p.name.startswith(".") or "baseline" in p.name.lower():
            continue
        candidate = p / csv_name
        if candidate.exists():
            csv_paths.append(candidate)
    return sorted(csv_paths)

def infer_sample_label(folder_name: str) -> str:
    """
    Returns labels like 'sample1', 'sample2', ... from folder name.
    Falls back to the full folder name if no match.
    """
    m = SAMPLE_REGEX.search(folder_name)
    if m:
        return f"sample{m.group(1)}"
    return folder_name  # fallback

def main():
    csv_files = list_sample_csvs(ROOT)
    if not csv_files:
        raise FileNotFoundError(f"No '{FEATURE_CSV_NAME}' files found under {ROOT}")

    frames = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        label = infer_sample_label(fp.parent.name)       # parent is the sample folder
        df["sample_label"] = label
        df["sample_folder"] = fp.parent.name             # keep provenance if useful
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(csv_files)} files → total rows: {len(all_df)}")

    # Reorder columns to show sample info first
    display_cols = (
        ["sample_label", "sample_folder"]
        + [c for c in all_df.columns if c not in ("sample_label", "sample_folder")]
    )
    all_df = all_df[display_cols]

    # Save the concatenated table
    out_path = ROOT / "all_events_labeled.csv"
    all_df.to_csv(out_path, index=False)
    print(f"Saved combined CSV to {out_path}")


if __name__ == "__main__":
    main()
