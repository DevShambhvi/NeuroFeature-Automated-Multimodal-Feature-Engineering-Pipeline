"""
NeuroFeature: Model Accuracy Evaluation

Evaluates the quality of extracted multimodal features by training
downstream classifiers and reporting per-modality accuracy metrics.

Metrics reported:
  - Confusion Matrix (Performance Matrix)
  - Classification Report (Precision, Recall, F1-Score)
  - ROC-AUC Score
  - Overall Accuracy

Modality ablation:
  Tests each feature subset independently (Tabular, Image, Audio, Fused)
  to measure the contribution of each modality.

  When provenance columns (has_tabular, has_image, has_audio) are present
  in the dataset, evaluation uses them to filter samples per modality —
  only samples that genuinely have features for a given modality are included
  in that modality's evaluation. This avoids polluting results with
  zero-padded vectors from missing modalities.

Usage:
    python neurofeatures/evaluate.py
    python neurofeatures/evaluate.py --data_path neurofeatures/outputs/neurofeature_final_dataset.csv
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except (ImportError, OSError) as e:
    HAS_TORCH = False
    from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# Expected embedding dimensions (must match assemble.py)
DIM_IMAGE = 1280   # MobileNetV2
DIM_AUDIO = 768    # Wav2Vec2 / librosa


# -----------------------------------------------------------------------
# PyTorch MLP Classifier wrapper for scikit-learn CV pipeline
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# PyTorch MLP Classifier wrapper for scikit-learn CV pipeline
# -----------------------------------------------------------------------
if HAS_TORCH:
    class PyTorchMLPClassifier:
        """Deeper MLP with BatchNorm, LR scheduling, and weight decay.

        Designed for Bolasso-selected + PCA-reduced fused features where a
        non-linear model can exploit interactions across modalities.
        """
        def __init__(
            self,
            input_dim=None,
            hidden_dims=None,
            dropout=0.3,
            epochs=80,
            lr=0.001,
            weight_decay=1e-4,
            batch_size=64,
        ):
            self.input_dim = input_dim
            self.hidden_dims = hidden_dims or [512, 256, 128]
            self.dropout = dropout
            self.epochs = epochs
            self.lr = lr
            self.weight_decay = weight_decay
            self.batch_size = batch_size
            self.model = None
            self.classes_ = None

        def _build_model(self, input_dim, num_classes):
            layers = []
            prev_dim = input_dim
            for i, h_dim in enumerate(self.hidden_dims):
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                # Slightly less dropout on the last hidden layer
                drop_rate = self.dropout if i < len(self.hidden_dims) - 1 else self.dropout * 0.67
                layers.append(nn.Dropout(drop_rate))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, num_classes))
            return nn.Sequential(*layers)

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            num_classes = len(self.classes_)
            input_dim = self.input_dim if self.input_dim is not None else X.shape[1]

            self.model = self._build_model(input_dim, num_classes)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=8, factor=0.5, min_lr=1e-6
            )

            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_mapped = np.searchsorted(self.classes_, y)
            y_tensor = torch.tensor(y_mapped, dtype=torch.long)

            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.epochs):
                self.model.train()
                epoch_loss = 0.0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / max(len(loader), 1)
                scheduler.step(avg_loss)
            return self

        def predict(self, X):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                return self.classes_[predicted.numpy()]

        def predict_proba(self, X):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                return probabilities.numpy()

        def get_params(self, deep=True):
            return {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "epochs": self.epochs,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
            }

        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self
else:
    # Dummy class placeholder if torch is missing or fails
    class PyTorchMLPClassifier:
        pass


# -----------------------------------------------------------------------
# Helper: detect which samples actually have non-zero features
# -----------------------------------------------------------------------
def _has_signal(X: np.ndarray, threshold: float = 1e-10) -> np.ndarray:
    """Return boolean mask: True for rows where at least one feature != 0."""
    return np.any(np.abs(X) > threshold, axis=1)


# -----------------------------------------------------------------------
# Core evaluation function
# -----------------------------------------------------------------------
def evaluate_modality(
    X: np.ndarray,
    y: np.ndarray,
    modality_name: str,
    n_splits: int = 5,
    config: Optional[dict] = None,
) -> Optional[dict]:
    """
    Train classifiers on feature subset X with labels y using
    stratified cross-validation.  Returns a dict of metrics or None
    if there is insufficient data.
    """
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        logger.warning(
            f"[{modality_name}] Fewer than 2 classes "
            f"({len(unique_classes)} class). Skipping."
        )
        return None

    n_samples = len(y)
    min_class_count = min(np.bincount(y.astype(int)))

    if n_samples < 6 or min_class_count < 2:
        logger.warning(
            f"[{modality_name}] Not enough samples for cross-validation "
            f"(n={n_samples}, min_class={min_class_count}). Skipping."
        )
        return None

    # Adjust n_splits if needed
    effective_splits = min(n_splits, min_class_count)
    if effective_splits < 2:
        effective_splits = 2

    logger.info(
        f"[{modality_name}] Evaluating {n_samples} samples, "
        f"{len(unique_classes)} classes, {effective_splits}-fold CV"
    )

    skf = StratifiedKFold(
        n_splits=effective_splits, shuffle=True, random_state=42
    )

    # --- Model 1: Random Forest (tuned) ---
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=5,
            random_state=42, n_jobs=-1,
        )),
    ])

    # --- Model 2: HistGradientBoosting (gold standard for tabular + embeddings) ---
    gb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=6,
            min_samples_leaf=10, random_state=42,
        )),
    ])

    # --- Model 3: Neural Network (Bolasso + NN combination) ---
    learned_cfg = {}
    if config:
        learned_cfg = config.get("fusion", {}).get("learned", {})

    # Support both old "hidden_dim" (int) and new "hidden_dims" (list) keys
    hidden_dims = learned_cfg.get("hidden_dims", None)
    if hidden_dims is None:
        hd = learned_cfg.get("hidden_dim", 128)
        hidden_dims = [hd]
    dropout = learned_cfg.get("dropout", 0.3)
    epochs = learned_cfg.get("epochs", 80)
    learning_rate = learned_cfg.get("learning_rate", 0.001)
    weight_decay = learned_cfg.get("weight_decay", 1e-4)
    batch_size = learned_cfg.get("batch_size", 64)

    if HAS_TORCH:
        nn_classifier = PyTorchMLPClassifier(
            input_dim=X.shape[1],
            hidden_dims=hidden_dims,
            dropout=dropout,
            epochs=epochs,
            lr=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )
    else:
        logger.warning(
            "PyTorch unavailable. Falling back to scikit-learn MLPClassifier."
        )
        from sklearn.neural_network import MLPClassifier
        nn_classifier = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_dims),
            max_iter=epochs * 100,
            learning_rate_init=learning_rate,
            random_state=42,
        )

    nn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", nn_classifier),
    ])

    results = {}

    for model_name, pipe in [
        ("RandomForest", rf_pipe),
        ("GradientBoosting", gb_pipe),
        ("NeuralNetwork", nn_pipe),
    ]:
        try:
            # Cross-val hard predictions
            y_pred = cross_val_predict(
                pipe, X, y, cv=skf, method="predict"
            )

            # Cross-val probability predictions (for ROC-AUC)
            try:
                y_prob = cross_val_predict(
                    pipe, X, y, cv=skf, method="predict_proba"
                )
            except Exception:
                y_prob = None

            # --- Metrics ---
            acc = accuracy_score(y, y_pred)
            cm = confusion_matrix(y, y_pred)
            cr = classification_report(
                y, y_pred, output_dict=True, zero_division=0
            )
            cr_text = classification_report(
                y, y_pred, zero_division=0
            )

            # ROC-AUC
            roc_auc = None
            if y_prob is not None:
                try:
                    if len(unique_classes) == 2:
                        roc_auc = roc_auc_score(y, y_prob[:, 1])
                    else:
                        roc_auc = roc_auc_score(
                            y, y_prob, multi_class="ovr", average="macro"
                        )
                except Exception as e:
                    logger.warning(
                        f"[{modality_name}/{model_name}] ROC-AUC failed: {e}"
                    )

            results[model_name] = {
                "accuracy": round(acc, 4),
                "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
                "confusion_matrix": cm.tolist(),
                "classification_report": cr,
                "classification_report_text": cr_text,
                "n_samples": int(n_samples),
                "n_classes": int(len(unique_classes)),
                "cv_folds": int(effective_splits),
            }

        except Exception as e:
            logger.error(f"[{modality_name}/{model_name}] Evaluation failed: {e}")
            results[model_name] = {"error": str(e)}

    return results


# -----------------------------------------------------------------------
# Pretty-print results
# -----------------------------------------------------------------------
def print_results(all_results: dict):
    """Print a clean summary table and detailed per-modality reports."""

    separator = "=" * 70
    thin_sep  = "-" * 70

    print(f"\n{separator}")
    print("  NEUROFEATURE — ACCURACY EVALUATION RESULTS")
    print(f"{separator}\n")

    # ---- Summary table ----
    print(f"{'Modality':<16} {'Model':<22} {'Accuracy':>10} {'ROC-AUC':>10} {'Samples':>8}")
    print(thin_sep)

    for modality, modality_results in all_results.items():
        if modality_results is None:
            print(f"{modality:<16} {'—':<22} {'SKIPPED':>10} {'—':>10} {'—':>8}")
            continue
        for model_name, metrics in modality_results.items():
            if "error" in metrics:
                print(f"{modality:<16} {model_name:<22} {'ERROR':>10} {'—':>10} {'—':>8}")
                continue
            acc_str = f"{metrics['accuracy']:.4f}"
            auc_str = (
                f"{metrics['roc_auc']:.4f}"
                if isinstance(metrics["roc_auc"], float)
                else metrics["roc_auc"]
            )
            n_str = str(metrics["n_samples"])
            print(f"{modality:<16} {model_name:<22} {acc_str:>10} {auc_str:>10} {n_str:>8}")

    print(f"{separator}\n")

    # ---- Detailed per-modality reports ----
    for modality, modality_results in all_results.items():
        if modality_results is None:
            continue

        print(f"\n{'─' * 70}")
        print(f"  DETAILED REPORT: {modality.upper()}")
        print(f"{'─' * 70}")

        for model_name, metrics in modality_results.items():
            if "error" in metrics:
                print(f"\n  [{model_name}] ERROR: {metrics['error']}")
                continue

            print(f"\n  ┌─ {model_name}")
            print(f"  │  Accuracy : {metrics['accuracy']:.4f}")
            print(f"  │  ROC-AUC  : {metrics['roc_auc']}")
            print(f"  │  Samples  : {metrics['n_samples']}")
            print(f"  │  CV Folds : {metrics['cv_folds']}")

            # Confusion Matrix
            cm = np.array(metrics["confusion_matrix"])
            print(f"  │")
            print(f"  │  Performance Matrix (Confusion Matrix):")
            # Header
            n_cls = cm.shape[0]
            header = "  │  " + " " * 12 + "  ".join(
                [f"Pred={c}" for c in range(n_cls)]
            )
            print(header)
            for i in range(n_cls):
                row_vals = "  ".join([f"{v:>6}" for v in cm[i]])
                print(f"  │  Actual={i}    {row_vals}")

            # Classification Report
            print(f"  │")
            print(f"  │  Classification Report:")
            for line in metrics["classification_report_text"].strip().split("\n"):
                print(f"  │    {line}")

            print(f"  └{'─' * 50}")

    print()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(
        description="Evaluate NeuroFeature model accuracy per modality"
    )
    parser.add_argument(
        "--data_path", default=None,
        help="Path to final fused dataset (CSV or Parquet)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to pipeline config file",
    )
    parser.add_argument(
        "--output_json", default=None,
        help="Path to save evaluation metrics as JSON",
    )
    parser.add_argument(
        "--cv_folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    args = parser.parse_args()

    # ---- Load config ----
    config = {}
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    paths = config.get("paths", {})
    output_dir = paths.get("output_dir", "neurofeatures/outputs")

    # ---- Resolve data path ----
    data_path = args.data_path
    if data_path is None:
        # Try Parquet first, then CSV
        parquet_path = os.path.join(output_dir, "neurofeature_final_dataset.parquet")
        csv_path = os.path.join(output_dir, "neurofeature_final_dataset.csv")
        if os.path.isfile(parquet_path):
            data_path = parquet_path
        elif os.path.isfile(csv_path):
            data_path = csv_path
        else:
            logger.error("No final dataset found. Run the pipeline first.")
            sys.exit(1)

    logger.info(f"Loading dataset from: {data_path}")

    # ---- Load dataset ----
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    logger.info(f"Dataset shape: {df.shape}")

    # ---- Check for provenance columns ----
    has_provenance = all(
        col in df.columns for col in ['has_tabular', 'has_image', 'has_audio']
    )
    if has_provenance:
        logger.info(
            "Provenance columns detected — using modality-aware evaluation. "
            f"Tabular: {df['has_tabular'].sum()}, "
            f"Image: {df['has_image'].sum()}, "
            f"Audio: {df['has_audio'].sum()}"
        )
    else:
        logger.info("No provenance columns found — falling back to zero-detection heuristic.")

    # ---- Extract labels ----
    if "label" not in df.columns:
        logger.error("No 'label' column found in dataset. Cannot evaluate.")
        sys.exit(1)

    # Drop samples without labels
    df_labeled = df.dropna(subset=["label"]).copy()
    logger.info(f"Labeled samples: {len(df_labeled)} / {len(df)}")

    if len(df_labeled) < 4:
        logger.error(
            f"Only {len(df_labeled)} labeled samples. Need at least 4 "
            f"for cross-validation. Add more labeled data to metadata."
        )
        sys.exit(1)

    # Encode labels
    y_raw = df_labeled["label"].values
    try:
        y_all = pd.to_numeric(y_raw, errors="raise").astype(int).values
    except Exception:
        le = LabelEncoder()
        y_all = le.fit_transform(y_raw.astype(str))

    # ---- Extract feature columns ----
    feat_cols = [c for c in df_labeled.columns if c.startswith("fused_feat_")]
    feat_cols.sort(key=lambda c: int(c.split("_")[-1]))

    X_all = df_labeled[feat_cols].values.astype(np.float32)
    D_total = X_all.shape[1]

    # ---- Load fusion metadata for dynamic dimensions ----
    fusion_meta_path = os.path.join(output_dir, "fusion_metadata.json")
    fusion_meta = {}
    if os.path.isfile(fusion_meta_path):
        with open(fusion_meta_path, "r") as f:
            fusion_meta = json.load(f)
        logger.info(f"Loaded fusion metadata: {fusion_meta}")
    else:
        logger.info("No fusion_metadata.json found — using hardcoded dimension constants.")

    logger.info(f"Total fused feature dimensions: {D_total}")

    # ---- Slice features by modality (dynamic dimensions) ----
    dim_image_actual = fusion_meta.get("dim_image", DIM_IMAGE)
    dim_audio_actual = fusion_meta.get("dim_audio", DIM_AUDIO)
    D_tab = fusion_meta.get("dim_tabular", D_total - dim_image_actual - dim_audio_actual)

    if D_tab < 0:
        logger.warning(
            f"Feature dimension mismatch: total={D_total}, "
            f"expected at least {dim_image_actual + dim_audio_actual}. "
            f"Treating all features as a single block."
        )
        D_tab = D_total
        dim_image_actual = 0
        dim_audio_actual = 0

    logger.info(
        f"Feature slicing: Tabular={D_tab}, "
        f"Image={dim_image_actual}, Audio={dim_audio_actual}"
    )

    # ---- Evaluate each modality ----
    all_results = {}

    # Define modality evaluation specs: (name, column slice, provenance column)
    modality_specs = [
        ("Tabular", (0, D_tab), "has_tabular"),
        ("Image", (D_tab, D_tab + dim_image_actual), "has_image"),
        ("Audio", (D_tab + dim_image_actual, D_total), "has_audio"),
    ]

    for modality_name, (col_start, col_end), prov_col in modality_specs:
        if col_start >= col_end:
            logger.info(f"[{modality_name}] No features available (dim=0). Skipping.")
            all_results[modality_name] = None
            continue

        X_mod = X_all[:, col_start:col_end]

        # Filter samples using provenance or zero-detection fallback
        if has_provenance and prov_col in df_labeled.columns:
            # Use provenance: only evaluate samples that genuinely have this modality
            prov_mask = df_labeled[prov_col].values.astype(bool)
            valid_mask = prov_mask
        else:
            # Fallback: use non-zero detection
            valid_mask = _has_signal(X_mod)

        X_valid = X_mod[valid_mask]
        y_valid = y_all[valid_mask]

        if len(y_valid) == 0:
            logger.info(f"[{modality_name}] No valid labeled samples. Skipping.")
            all_results[modality_name] = None
            continue

        logger.info(
            f"[{modality_name}] {valid_mask.sum()} samples with real features "
            f"(of {len(valid_mask)} labeled total)"
        )

        result = evaluate_modality(
            X_valid, y_valid, modality_name, n_splits=args.cv_folds, config=config
        )
        all_results[modality_name] = result

    # ---- Fused evaluation ----
    # For fused: include samples that have at least one modality with real features
    if has_provenance:
        fused_mask = (
            df_labeled['has_tabular'].values.astype(bool) |
            df_labeled['has_image'].values.astype(bool) |
            df_labeled['has_audio'].values.astype(bool)
        )
    else:
        fused_mask = _has_signal(X_all)

    X_fused = X_all[fused_mask]
    y_fused = y_all[fused_mask]

    if len(y_fused) > 0:
        logger.info(f"[Fused] {fused_mask.sum()} samples with at least one modality")
        all_results["Fused"] = evaluate_modality(
            X_fused, y_fused, "Fused", n_splits=args.cv_folds, config=config
        )
    else:
        logger.info("[Fused] No valid labeled samples. Skipping.")
        all_results["Fused"] = None

    # ---- Print results ----
    print_results(all_results)

    # ---- Save JSON report (optional) ----
    json_path = args.output_json or os.path.join(
        output_dir, "evaluation_metrics.json"
    )

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {}
    for mod, res in all_results.items():
        if res is None:
            serializable[mod] = None
            continue
        serializable[mod] = {}
        for model_name, metrics in res.items():
            serializable[mod][model_name] = {
                k: _convert(v) for k, v in metrics.items()
                if k != "classification_report_text"
            }

    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=_convert)
    logger.info(f"Metrics saved to: {json_path}")


if __name__ == "__main__":
    main()
