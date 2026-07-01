"""
neurofeature/ingest.py

Scans data/text, data/images, and data/audio folders independently.
Each modality gets its own metadata CSV. No cross-modality merging.

Output files:
    outputs/indexes/text_metadata.csv
    outputs/indexes/image_metadata.csv
    outputs/indexes/audio_metadata.csv

Usage (from repo root):
    python neurofeature/ingest.py
    python neurofeature/ingest.py --data_dir data/
    python neurofeature/ingest.py --label_file labels.csv
"""

import os
import argparse
from typing import Optional
import yaml

import pandas as pd


# ---------------------------------------------------------------------------
# Supported extensions per modality
# ---------------------------------------------------------------------------
TEXT_EXTS  = {".txt", ".md", ".json", ".csv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ---------------------------------------------------------------------------
# Scan a single folder
# ---------------------------------------------------------------------------
def scan_folder(folder_path: str, valid_exts: set, modality: str) -> pd.DataFrame:
    """
    Walk a folder and return a DataFrame:
        sample_id      — filename without extension
        <modality>_path — relative path to the file
        modality       — which modality this row belongs to
    """
    records = []

    if not os.path.isdir(folder_path):
        print(f"  [warn] folder not found, skipping: {folder_path}")
        return pd.DataFrame(columns=["sample_id", f"{modality}_path", "label"])

    for fname in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_exts:
            continue
        sample_id = os.path.splitext(fname)[0]
        file_path = os.path.join(folder_path, fname)
        records.append({
            "sample_id": sample_id,
            f"{modality}_path": file_path,
            "label": None        # filled in later if label_file provided
        })

    df = pd.DataFrame(records)
    print(f"  [ok]   {folder_path}: {len(df)} files found")
    return df


# ---------------------------------------------------------------------------
# Attach labels from a CSV (optional)
# ---------------------------------------------------------------------------
def attach_labels(df: pd.DataFrame, label_file: str, modality: str) -> pd.DataFrame:
    """
    label_file must have columns: sample_id, label
    Left-joins onto the modality DataFrame.
    """
    labels = pd.read_csv(label_file)
    if "sample_id" not in labels.columns or "label" not in labels.columns:
        raise ValueError(
            f"Label file must have 'sample_id' and 'label' columns. "
            f"Found: {list(labels.columns)}"
        )
    df = df.drop(columns=["label"], errors="ignore")
    df = df.merge(labels, on="sample_id", how="left")
    print(f"  [labels] attached to {modality} index from {label_file}")
    return df


# ---------------------------------------------------------------------------
# Save a metadata CSV
# ---------------------------------------------------------------------------
def save_metadata(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [saved] {path}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(
    config_path: str = "config.yaml",
    tabular_dir: Optional[str] = None,
    image_dir: Optional[str] = None,
    audio_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    label_file: Optional[str] = None,
) -> dict:
    """
    Independently scan each modality folder and produce separate metadata CSVs.

    Returns a dict with keys: 'tabular', 'image', 'audio'
    Each value is the corresponding DataFrame.
    """
    # Load config if exists
    config = {}
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
    paths = config.get("paths", {})
    cfg_tabular = paths.get("tabular_input_dir", "neurofeatures/data/tabular")
    cfg_image   = paths.get("image_dir",         "neurofeatures/data/images")
    cfg_audio   = paths.get("audio_dir",         "neurofeatures/data/audio")
    cfg_index   = paths.get("index_dir",         "neurofeatures/outputs/indexes")
    
    t_dir = tabular_dir or cfg_tabular
    i_dir = image_dir or cfg_image
    a_dir = audio_dir or cfg_audio
    out_dir = output_dir or cfg_index

    print("\n=== NeuroFeature: ingest (independent modalities) ===")
    print(f"  tabular dir: {t_dir}")
    print(f"  image dir  : {i_dir}")
    print(f"  audio dir  : {a_dir}")
    print(f"  output dir : {out_dir}\n")

    modalities = {
        "tabular": (t_dir, TEXT_EXTS),
        "image":   (i_dir, IMAGE_EXTS),
        "audio":   (a_dir, AUDIO_EXTS),
    }

    results = {}

    for modality, (folder, exts) in modalities.items():
        df = scan_folder(folder, exts, modality)

        if label_file and os.path.isfile(label_file):
            df = attach_labels(df, label_file, modality)

        out_path = os.path.join(out_dir, f"{modality}_metadata.csv")
        save_metadata(df, out_path)
        results[modality] = df

    # Summary
    print("\n=== Summary =========================================")
    for modality, df in results.items():
        path_col = f"{modality}_path"
        count = df[path_col].notna().sum() if path_col in df.columns else 0
        print(f"  {modality:<8} : {count} samples")
    print("====================================================\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan data folders independently and generate per-modality metadata CSVs"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to pipeline configuration file"
    )
    parser.add_argument(
        "--tabular_dir", default=None,
        help="Tabular input directory (overrides config)"
    )
    parser.add_argument(
        "--image_dir", default=None,
        help="Image input directory (overrides config)"
    )
    parser.add_argument(
        "--audio_dir", default=None,
        help="Audio input directory (overrides config)"
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to save metadata CSVs (overrides config)"
    )
    parser.add_argument(
        "--label_file", default=None,
        help="Optional CSV with sample_id and label columns"
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        tabular_dir=args.tabular_dir,
        image_dir=args.image_dir,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        label_file=args.label_file,
    )