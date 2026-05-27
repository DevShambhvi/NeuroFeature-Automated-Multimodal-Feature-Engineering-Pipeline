"""
neurofeature/extract/tabular.py

Processes independent numeric CSV datasets.
Each CSV is handled separately since they have different column structures.

For each CSV:
    1. Load and validate
    2. Separate features from label column (if present)
    3. Handle missing values
    4. Scale numeric features
    5. Export a clean feature CSV ready for downstream use

Usage (from repo root):
    python neurofeatures/extract/tabular.py
    python neurofeatures/extract/tabular.py --input_dir neurofeatures/data/text/
"""

import os
import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Supported extensions and label candidates
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = {".csv"}
DEFAULT_LABEL_CANDIDATES = {"label", "target", "class", "y", "output"}

# All encodings to try in order — covers virtually every real-world CSV
ENCODINGS = [
    "utf-8",
    "utf-8-sig",   # UTF-8 with BOM
    "latin-1",     # iso-8859-1, never fails (single-byte fallback)
    "cp1252",      # Windows Western European
    "iso-8859-1",
    "iso-8859-2",
    "cp1250",      # Windows Central European
    "cp1251",      # Windows Cyrillic
    "mac_roman",
]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    """Load pipeline config from YAML file."""
    if not os.path.isfile(config_path):
        print(f"  [warn] config.yaml not found at {config_path}, using CLI args")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_paths_from_config(config: dict) -> tuple:
    """Extract input_dir and output_dir from config paths section."""
    paths = config.get("paths", {})
    input_dir  = paths.get("text_dir",    "data/text")
    output_dir = paths.get("tabular_dir", "outputs/tabular")
    return input_dir, output_dir


def get_extraction_settings(config: dict) -> dict:
    """Extract tabular extraction settings from config."""
    return config.get("extraction", {}).get("text", {})


# ---------------------------------------------------------------------------
# Step 1 — Load and validate
# ---------------------------------------------------------------------------
def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV trying multiple encodings in order.
    Falls back to errors='replace' as absolute last resort
    so no file ever causes a hard crash on encoding.
    """
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
            if encoding != "utf-8":
                print(f"  encoding : {encoding} (fallback)")
            break
        except UnicodeDecodeError:
            continue
    else:
        # Absolute last resort — replace undecodable bytes with ?
        print(f"  encoding : utf-8 with replacement (last resort)")
        df = pd.read_csv(
            file_path,
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace",
        )

    print(f"\n  file     : {os.path.basename(file_path)}")
    print(f"  shape    : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  columns  : {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Detect and separate label column
# ---------------------------------------------------------------------------
def separate_label(df: pd.DataFrame, label_col=None):
    """
    Separate label column from feature columns.
    Auto-detects label column if not specified.
    Returns (feature_df, label_series or None).
    """
    if label_col and label_col in df.columns:
        print(f"  label    : '{label_col}' (specified)")
        return df.drop(columns=[label_col]), df[label_col]

    for col in df.columns:
        if col.lower() in DEFAULT_LABEL_CANDIDATES:
            print(f"  label    : '{col}' (auto-detected)")
            return df.drop(columns=[col]), df[col]

    print(f"  label    : none detected — unsupervised mode")
    return df, None


# ---------------------------------------------------------------------------
# Step 3 — Keep only numeric columns
# ---------------------------------------------------------------------------
def keep_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-numeric columns (IDs, strings, dates)."""
    numeric_df = df.select_dtypes(include=[np.number])
    dropped = set(df.columns) - set(numeric_df.columns)
    if dropped:
        print(f"  dropped  : {sorted(dropped)} (non-numeric)")
    print(f"  features : {len(numeric_df.columns)} numeric columns kept")
    return numeric_df


# ---------------------------------------------------------------------------
# Step 4 — Handle missing values
# ---------------------------------------------------------------------------
def impute_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Fill missing values. Drops unnamed/all-NaN columns first."""
    unnamed   = [c for c in df.columns if str(c).startswith("Unnamed")]
    all_nan   = [c for c in df.columns if df[c].isnull().all()]
    drop_cols = list(set(unnamed + all_nan))
    if drop_cols:
        print(f"  dropped  : {sorted(drop_cols)} (empty/unnamed columns)")
        df = df.drop(columns=drop_cols)

    missing = df.isnull().sum().sum()
    if missing == 0:
        print(f"  missing  : none")
        return df

    print(f"  missing  : {missing} values → imputing with '{strategy}'")
    imputer = SimpleImputer(strategy=strategy)
    imputed = imputer.fit_transform(df)
    return pd.DataFrame(imputed, columns=df.columns)


# ---------------------------------------------------------------------------
# Step 5 — Scale features
# ---------------------------------------------------------------------------
def scale_features(df: pd.DataFrame) -> tuple:
    """Standardize to zero mean and unit variance."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns)
    print(f"  scaling  : StandardScaler applied")
    return scaled_df, scaler


# ---------------------------------------------------------------------------
# Step 6 — Clip extreme outliers
# ---------------------------------------------------------------------------
def clip_outliers(df: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
    """Clip values beyond n_std standard deviations after scaling."""
    clipped   = df.clip(lower=-n_std, upper=n_std)
    n_clipped = (df.abs() > n_std).sum().sum()
    if n_clipped > 0:
        print(f"  outliers : {n_clipped} values clipped beyond ±{n_std}σ")
    return clipped


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------
def save_features(
    feature_df: pd.DataFrame,
    label_series,
    output_path: str,
) -> None:
    """Save scaled features (+ label if present) to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df = feature_df.copy()
    if label_series is not None:
        out_df["label"] = label_series.values
    out_df.to_csv(output_path, index=False)
    print(f"  saved    : {output_path}")


# ---------------------------------------------------------------------------
# Process a single CSV file
# ---------------------------------------------------------------------------
def process_file(
    file_path: str,
    output_dir: str,
    label_col=None,
    impute_strategy: str = "mean",
    clip_std: float = 3.0,
) -> pd.DataFrame:
    """Full pipeline for one CSV: load → label → numeric → impute → scale → clip → save."""
    print("\n" + "─" * 52)

    df = load_csv(file_path)
    feature_df, label_series = separate_label(df, label_col)
    feature_df = keep_numeric(feature_df)

    if feature_df.empty:
        print(f"  [skip] no numeric columns found in {file_path}")
        return pd.DataFrame()

    feature_df = impute_missing(feature_df, strategy=impute_strategy)

    if feature_df.empty:
        print(f"  [skip] no columns remaining after cleaning in {file_path}")
        return pd.DataFrame()

    feature_df, _ = scale_features(feature_df)
    feature_df    = clip_outliers(feature_df, n_std=clip_std)

    base_name   = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_features.csv")
    save_features(feature_df, label_series, output_path)

    return feature_df


# ---------------------------------------------------------------------------
# Process all CSVs in a folder
# ---------------------------------------------------------------------------
def run(
    input_dir: str = "data/text",
    output_dir: str = "outputs/tabular",
    label_col=None,
    impute_strategy: str = "mean",
    clip_std: float = 3.0,
) -> dict:
    """Process all CSV files in input_dir independently."""
    print("\n── NeuroFeature: tabular extraction ────────────────")
    print(f"  input dir  : {input_dir}")
    print(f"  output dir : {output_dir}")

    if not os.path.isdir(input_dir):
        print(f"  [error] input directory not found: {input_dir}")
        return {}

    csv_files = [
        f for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]

    if not csv_files:
        print(f"  [warn] no CSV files found in {input_dir}")
        return {}

    print(f"  found      : {len(csv_files)} CSV file(s)\n")

    results = {}
    failed  = []

    for fname in csv_files:
        file_path = os.path.join(input_dir, fname)
        try:
            result_df = process_file(
                file_path=file_path,
                output_dir=output_dir,
                label_col=label_col,
                impute_strategy=impute_strategy,
                clip_std=clip_std,
            )
            results[fname] = result_df
        except Exception as e:
            print(f"  [error] {fname}: {e}")
            failed.append((fname, str(e)))

    # Summary
    processed = [f for f, df in results.items() if not df.empty]
    skipped   = [f for f, df in results.items() if df.empty]

    print("\n── Summary ─────────────────────────────────────────")
    print(f"  total     : {len(csv_files)} files")
    print(f"  processed : {len(processed)}")
    print(f"  skipped   : {len(skipped)} (no numeric columns)")
    print(f"  failed    : {len(failed)}")

    if skipped:
        print(f"\n  [skipped]")
        for f in skipped:
            print(f"    {f}")

    if failed:
        print(f"\n  [failed]")
        for f, err in failed:
            print(f"    {f}: {err}")

    print("────────────────────────────────────────────────────\n")

    return results


# ---------------------------------------------------------------------------
# Entry point — reads config.yaml, CLI args override
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract features from numeric CSV datasets independently"
    )
    parser.add_argument("--config",          default="config.yaml")
    parser.add_argument("--input_dir",       default=None,
        help="Overrides paths.text_dir in config.yaml")
    parser.add_argument("--output_dir",      default=None,
        help="Overrides paths.tabular_dir in config.yaml")
    parser.add_argument("--label_col",       default=None)
    parser.add_argument("--impute_strategy", default=None,
        choices=["mean", "median", "most_frequent"])
    parser.add_argument("--clip_std",        default=None, type=float)
    args = parser.parse_args()

    config               = load_config(args.config)
    cfg_input, cfg_output = get_paths_from_config(config)
    cfg_extract          = get_extraction_settings(config)

    run(
        input_dir       = args.input_dir       or cfg_input,
        output_dir      = args.output_dir      or cfg_output,
        label_col       = args.label_col       or cfg_extract.get("label_col"),
        impute_strategy = args.impute_strategy or cfg_extract.get("impute_strategy", "mean"),
        clip_std        = args.clip_std        or cfg_extract.get("clip_std", 3.0),
    )


if __name__ == "__main__":
    main()