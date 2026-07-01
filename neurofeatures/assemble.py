"""
NeuroFeature Pipeline: Dataset Assembly and Baseline Fusion

This module handles Stage 3 and 4 of the NeuroFeature pipeline.
It merges extracted modality features (Tabular, Image, Audio) based on `sample_id`,
applies the configured missing-modality policies, and performs baseline 
concatenation fusion to export a model-ready tabular dataset.

Supports three missing-modality policies:
  - 'skip':        Drop samples missing any enabled modality.
  - 'zero_vector': Pad missing modalities with zero vectors.
  - 'independent': Include all samples, pad missing modalities with zero vectors,
                    and record provenance flags (has_tabular, has_image, has_audio)
                    so downstream evaluation can filter per-modality correctly.
                    This is the recommended policy when modalities have
                    non-overlapping sample_id namespaces.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Literal, Optional
import argparse
import yaml

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Define expected embedding dimensions based on our extractors
DIM_TABULAR = 768  # Tabular statistics
DIM_IMAGE = 1280   # MobileNetV2
DIM_AUDIO = 768    # Wav2Vec2 Base


def auto_label_image(sample_id: str) -> Optional[int]:
    """Derive binary labels from image filenames (real vs fake).
    
    Returns 1 for 'real_*', 0 for 'fake_*', None otherwise.
    """
    sid = sample_id.lower()
    if sid.startswith("real"):
        return 1
    elif sid.startswith("fake"):
        return 0
    return None


class DatasetAssembler:
    def __init__(
        self, 
        metadata_path: str,
        missing_policy: Literal['skip', 'zero_vector', 'independent'] = 'independent',
        selection_config: Optional[dict] = None,
        pca_config: Optional[dict] = None,
    ):
        """
        Args:
            metadata_path: Path to the master CSV containing sample_ids and labels.
            missing_policy: How to handle samples missing a modality.
                - 'skip': Drop samples missing any modality.
                - 'zero_vector': Pad missing modalities with zero vectors.
                - 'independent': Like zero_vector but also records provenance flags
                  so evaluation can correctly filter per modality.
            selection_config: Dictionary containing Bolasso selection configurations.
            pca_config: Dictionary with per-modality normalization / PCA settings.
        """
        self.metadata_path = Path(metadata_path)
        self.missing_policy = missing_policy
        self.selection_config = selection_config or {"enabled": False}
        self.pca_config = pca_config or {"enabled": False}
        
        # Actual dimensions (may change after Bolasso / PCA; set during assemble)
        self.dim_tabular_actual = DIM_TABULAR
        self.dim_image_actual = DIM_IMAGE
        self.dim_audio_actual = DIM_AUDIO
        
        logger.info(f"Initializing Assembler | Policy: {self.missing_policy.upper()}")
        
        # Load metadata if available (may be empty or minimal)
        if self.metadata_path.exists():
            self.metadata_df = pd.read_csv(self.metadata_path)
            self.metadata_df['sample_id'] = self.metadata_df['sample_id'].astype(str)
        else:
            logger.warning(f"Metadata file not found: {self.metadata_path}. Will discover samples from NPZ files.")
            self.metadata_df = pd.DataFrame(columns=['sample_id', 'label'])

    def _load_npz_features(self, npz_path: Path, expected_dim: int) -> Dict[str, np.ndarray]:
        """Loads an NPZ file and maps the base filename to its feature array."""
        if not npz_path.exists():
            logger.warning(f"Feature file not found: {npz_path}")
            return {}
            
        data = np.load(npz_path)
        features = data['features']
        paths = data['paths']
        
        # Extract the base filename (without extension) to match with sample_id
        feature_dict = {}
        for path_str, feat_vec in zip(paths, features):
            base_name = Path(path_str).stem 
            feature_dict[base_name] = feat_vec
            
        logger.info(f"Loaded {len(feature_dict)} features from {npz_path.name}")
        return feature_dict

    def _normalize_and_reduce(
        self,
        feats_dict: Dict[str, np.ndarray],
        modality_name: str,
        n_components: Optional[int] = None,
    ) -> tuple:
        """Apply StandardScaler and optional PCA to a modality's feature dict.

        Args:
            feats_dict: Mapping sample_id → feature vector. Modified in-place.
            modality_name: Name for logging.
            n_components: If set, apply PCA to this many components.

        Returns:
            (feats_dict, actual_dim) where actual_dim is the output dimensionality.
        """
        if not feats_dict:
            return feats_dict, 0

        keys = list(feats_dict.keys())
        matrix = np.array([feats_dict[k] for k in keys], dtype=np.float32)

        # Sanitize NaN / Inf
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        original_dim = matrix.shape[1]

        # StandardScaler normalisation
        scaler = StandardScaler()
        matrix = scaler.fit_transform(matrix)

        # Optional PCA
        if n_components is not None and n_components > 0:
            n_comp = min(n_components, matrix.shape[1], matrix.shape[0])
            if n_comp >= 1:
                pca = PCA(n_components=n_comp, random_state=42)
                matrix = pca.fit_transform(matrix)
                explained = sum(pca.explained_variance_ratio_) * 100
                logger.info(
                    f"[{modality_name}] PCA: {original_dim} → {n_comp} components "
                    f"({explained:.1f}% variance retained)"
                )

        actual_dim = matrix.shape[1]

        # Write transformed vectors back into the dict
        for key, vec in zip(keys, matrix):
            feats_dict[key] = vec.astype(np.float32)

        logger.info(
            f"[{modality_name}] Normalised: {len(keys)} samples, "
            f"{original_dim} → {actual_dim} dims"
        )
        return feats_dict, actual_dim

    def _discover_all_sample_ids(
        self,
        tabular_feats: Dict[str, np.ndarray],
        image_feats: Dict[str, np.ndarray],
        audio_feats: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Build a unified sample registry from the union of metadata + all NPZ keys.
        
        Returns a DataFrame with columns: sample_id, label
        Each sample_id appears exactly once. Labels are resolved with priority:
          1. Metadata CSV label (if present and not NaN)
          2. Auto-label from image filename convention (real_* → 1, fake_* → 0)
          3. NaN (unlabeled)
        """
        # Start with metadata samples
        meta_labels = {}
        for _, row in self.metadata_df.iterrows():
            sid = str(row['sample_id'])
            label = row.get('label')
            if pd.notna(label):
                meta_labels[sid] = label
        
        # Collect all unique sample_ids
        all_ids = set(self.metadata_df['sample_id'].astype(str).tolist())
        all_ids.update(tabular_feats.keys())
        all_ids.update(image_feats.keys())
        all_ids.update(audio_feats.keys())
        
        logger.info(
            f"Sample discovery: metadata={len(self.metadata_df)}, "
            f"tabular_npz={len(tabular_feats)}, image_npz={len(image_feats)}, "
            f"audio_npz={len(audio_feats)} → union={len(all_ids)} unique samples"
        )
        
        # Build label lookup with priority: metadata > auto-label > NaN
        rows = []
        for sid in sorted(all_ids):
            label = meta_labels.get(sid)
            if label is None:
                # Try auto-labeling from image filename convention
                if sid in image_feats:
                    auto_lbl = auto_label_image(sid)
                    if auto_lbl is not None:
                        label = auto_lbl
            rows.append({'sample_id': sid, 'label': label})
        
        registry = pd.DataFrame(rows)
        labeled_count = registry['label'].notna().sum()
        logger.info(f"Label resolution: {labeled_count}/{len(registry)} samples have labels")
        return registry

    def _run_bolasso(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_bootstraps: int = 50, 
        alpha: float = 0.01, 
        threshold: float = 0.6
    ) -> np.ndarray:
        """
        Runs Bolasso (Bootstrap Lasso) to select features from handcrafted tabular features.
        """
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler
        
        N, D = X.shape
        logger.info(f"Running Bolasso on {N} samples with {D} features...")
        
        # Standardize features (handling constant columns)
        stds = np.std(X, axis=0)
        non_constant_indices = np.where(stds > 1e-8)[0]
        
        if len(non_constant_indices) == 0:
            logger.warning("All tabular features are constant. Bolasso cannot select features. Keeping all.")
            return np.ones(D, dtype=bool)
            
        X_filtered = X[:, non_constant_indices]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        # Track selection counts
        selection_counts = np.zeros(len(non_constant_indices))
        
        for i in range(n_bootstraps):
            # Bootstrap sample
            indices = np.random.choice(N, size=N, replace=True)
            X_b = X_scaled[indices]
            y_b = y[indices]
            
            # Check variance of target in bootstrap
            if np.std(y_b) < 1e-8:
                continue
                
            try:
                lasso = Lasso(alpha=alpha, max_iter=2000, random_state=i)
                lasso.fit(X_b, y_b)
                # Count non-zero coefficients
                non_zero = (np.abs(lasso.coef_) > 1e-5)
                selection_counts += non_zero
            except Exception as e:
                continue
                
        # Calculate selection frequencies
        selection_freq = selection_counts / n_bootstraps
        selected_filtered = (selection_freq >= threshold)
        
        # Reconstruct the full boolean mask for all D features
        selected_mask = np.zeros(D, dtype=bool)
        selected_mask[non_constant_indices] = selected_filtered
        
        # Fallback: if zero features selected, select all non-constant features to prevent empty vector
        if selected_mask.sum() == 0:
            logger.warning("Bolasso selected 0 features. Selecting all non-constant features as fallback.")
            selected_mask[non_constant_indices] = True
            
        logger.info(f"Bolasso: selected {selected_mask.sum()} / {D} features (threshold: {threshold})")
        return selected_mask

    def assemble(self, tabular_npz: str, image_npz: str, audio_npz: str) -> pd.DataFrame:
        """
        Merges all modalities using union-based sample discovery.
        
        Discovers sample_ids from the union of metadata CSV and all NPZ feature
        files, so non-overlapping modality namespaces are handled correctly.
        Records provenance flags (has_tabular, has_image, has_audio) to enable
        modality-aware downstream evaluation.

        Pipeline:
          1. Load raw NPZ features
          2. Per-modality StandardScaler + optional PCA (image & audio)
          3. Bolasso feature selection on tabular features
          4. Concatenation fusion with provenance tracking
        """
        logger.info("Loading extracted modality features...")
        tabular_feats = self._load_npz_features(Path(tabular_npz), DIM_TABULAR)
        image_feats = self._load_npz_features(Path(image_npz), DIM_IMAGE)
        audio_feats = self._load_npz_features(Path(audio_npz), DIM_AUDIO)

        # ------------------------------------------------------------------
        # Per-modality normalisation + PCA (before Bolasso / concatenation)
        # ------------------------------------------------------------------
        pca_enabled = self.pca_config.get("enabled", False)
        normalize = self.pca_config.get("normalize_per_modality", True)

        if image_feats and (normalize or pca_enabled):
            n_comp = self.pca_config.get("image_components") if pca_enabled else None
            image_feats, self.dim_image_actual = self._normalize_and_reduce(
                image_feats, "Image", n_components=n_comp,
            )

        if audio_feats and (normalize or pca_enabled):
            n_comp = self.pca_config.get("audio_components") if pca_enabled else None
            audio_feats, self.dim_audio_actual = self._normalize_and_reduce(
                audio_feats, "Audio", n_components=n_comp,
            )

        # Tabular: normalise only (Bolasso handles dimensionality reduction)
        if tabular_feats and normalize:
            tabular_feats, _ = self._normalize_and_reduce(
                tabular_feats, "Tabular", n_components=None,
            )

        # Discover all sample_ids from the union of metadata + NPZ files
        sample_registry = self._discover_all_sample_ids(tabular_feats, image_feats, audio_feats)
        
        assembled_rows = []
        skipped_count = 0
        
        # ------------------------------------------------------------------
        # Bolasso feature selection on tabular handcrafted statistics
        # ------------------------------------------------------------------
        selection_enabled = self.selection_config.get("enabled", False)
        selected_tabular_indices = None
        
        if selection_enabled:
            logger.info("Feature selection (Bolasso) is enabled for handcrafted tabular statistics.")
            X_tab_list = []
            valid_positions = []
            
            for pos, (_, row) in enumerate(sample_registry.iterrows()):
                sample_id = str(row['sample_id'])
                t_vec = tabular_feats.get(sample_id)
                if t_vec is None:
                    t_vec = np.zeros(DIM_TABULAR, dtype=np.float32)
                X_tab_list.append(t_vec)
                
                label = row.get('label')
                if pd.notna(label) and sample_id in tabular_feats:
                    valid_positions.append(pos)
                    
            X_tab = np.array(X_tab_list, dtype=np.float32)
            
            if len(valid_positions) >= 2:
                X_tab_valid = X_tab[valid_positions]
                labels_valid = sample_registry.iloc[valid_positions]['label']
                
                from sklearn.preprocessing import LabelEncoder
                try:
                    y_numeric = pd.to_numeric(labels_valid, errors='raise').values
                except Exception:
                    le = LabelEncoder()
                    y_numeric = le.fit_transform(labels_valid.astype(str))
                    
                n_bootstraps = self.selection_config.get("n_bootstraps", 100)
                alpha = self.selection_config.get("lasso_alpha", 0.01)
                threshold = self.selection_config.get("selection_threshold", 0.5)
                
                selected_tabular_indices = self._run_bolasso(
                    X=X_tab_valid,
                    y=y_numeric,
                    n_bootstraps=n_bootstraps,
                    alpha=alpha,
                    threshold=threshold
                )
                self.dim_tabular_actual = int(selected_tabular_indices.sum())
                logger.info(f"Bolasso mask shape: {selected_tabular_indices.shape}, "
                            f"selected: {self.dim_tabular_actual} / {DIM_TABULAR}")
            else:
                logger.warning("Not enough labeled tabular samples to run Bolasso. Skipping selection.")
                
        logger.info("Assembling multimodal dataset...")
        
        for _, row in sample_registry.iterrows():
            sample_id = str(row['sample_id'])
            
            # Try to lookup actual file paths from metadata for mapping stems
            metadata_row = self.metadata_df[self.metadata_df['sample_id'] == sample_id]
            
            t_key = sample_id
            i_key = sample_id
            a_key = sample_id
            
            if not metadata_row.empty:
                m_row = metadata_row.iloc[0]
                t_path = m_row.get('tabular_path')
                i_path = m_row.get('image_path')
                a_path = m_row.get('audio_path')
                
                if pd.notna(t_path) and str(t_path).strip() != "":
                    t_key = Path(t_path).stem
                if pd.notna(i_path) and str(i_path).strip() != "":
                    i_key = Path(i_path).stem
                if pd.notna(a_path) and str(a_path).strip() != "":
                    a_key = Path(a_path).stem
            
            # 1. Fetch features (returns array or None)
            t_vec = tabular_feats.get(t_key)
            i_vec = image_feats.get(i_key)
            a_vec = audio_feats.get(a_key)
            
            # Track provenance: which modalities have real features
            has_tabular = t_vec is not None
            has_image = i_vec is not None
            has_audio = a_vec is not None
            
            # 2. Apply missing-modality policy
            if self.missing_policy == 'skip':
                if t_vec is None or i_vec is None or a_vec is None:
                    skipped_count += 1
                    continue

            # 3. Resolve missing features with zero vectors (using actual dims)
            t_arr: np.ndarray = t_vec if t_vec is not None else np.zeros(DIM_TABULAR, dtype=np.float32)
            
            # Apply Bolasso selection if active
            if selected_tabular_indices is not None:
                t_arr = t_arr[selected_tabular_indices]
                
            i_arr: np.ndarray = i_vec if i_vec is not None else np.zeros(self.dim_image_actual, dtype=np.float32)
            a_arr: np.ndarray = a_vec if a_vec is not None else np.zeros(self.dim_audio_actual, dtype=np.float32)

            # 4. Baseline Fusion (Concatenation)
            fused_vector = np.concatenate((t_arr, i_arr, a_arr))
            
            # 5. Build the row dictionary with provenance flags
            row_data = {
                'sample_id': sample_id,
                'label': row.get('label', np.nan),
                'has_tabular': has_tabular,
                'has_image': has_image,
                'has_audio': has_audio,
            }
            
            # Add the concatenated features dynamically
            for idx, val in enumerate(fused_vector):
                row_data[f"fused_feat_{idx}"] = val
                
            assembled_rows.append(row_data)

        if self.missing_policy == 'skip':
            logger.info(f"Skipped {skipped_count} samples due to missing modalities.")
            
        final_df = pd.DataFrame(assembled_rows)
        
        # Log provenance summary
        if not final_df.empty and 'has_tabular' in final_df.columns:
            n_tab = final_df['has_tabular'].sum()
            n_img = final_df['has_image'].sum()
            n_aud = final_df['has_audio'].sum()
            n_multi = ((final_df['has_tabular'].astype(int) + 
                        final_df['has_image'].astype(int) + 
                        final_df['has_audio'].astype(int)) > 1).sum()
            logger.info(
                f"Provenance: tabular={n_tab}, image={n_img}, audio={n_aud}, "
                f"multi-modal={n_multi}"
            )
        
        logger.info(f"Assembly complete! Final Dataset Shape: {final_df.shape}")
        return final_df


def main():
    """CLI entry point for the NeuroFeature Dataset Assembler."""
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(
        description="Assemble multimodal datasets in the NeuroFeature pipeline"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--tabular_npz", default=None)
    parser.add_argument("--image_npz", default=None)
    parser.add_argument("--audio_npz", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            
    paths = config.get("paths", {})
    metadata_cfg = config.get("metadata", {})
    
    # Resolve paths
    data_root = paths.get("data_root", "neurofeatures")
    cfg_metadata = os.path.join(data_root, metadata_cfg.get("path", "metadata_example.csv"))
    
    cfg_tabular_feat = paths.get("tabular_feat_dir", "neurofeatures/outputs")
    cfg_image_feat   = paths.get("image_feat_dir",   "neurofeatures/outputs")
    cfg_audio_feat   = paths.get("audio_feat_dir",   "neurofeatures/outputs")
    cfg_output       = paths.get("output_dir",       "neurofeatures/outputs")
    
    metadata_path = args.metadata_path or cfg_metadata
    tabular_npz   = args.tabular_npz or os.path.join(cfg_tabular_feat, "tabular_features.npz")
    image_npz     = args.image_npz or os.path.join(cfg_image_feat, "image_features.npz")
    audio_npz     = args.audio_npz or os.path.join(cfg_audio_feat, "audio_features.npz")
    output_dir    = args.output_dir or cfg_output
    
    out_parquet = os.path.join(output_dir, "neurofeature_final_dataset.parquet")
    out_csv     = os.path.join(output_dir, "neurofeature_final_dataset.csv")
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found at {metadata_path}. Will discover samples from NPZ files only.")

    # 2. Initialize Assembler
    selection_cfg = config.get("selection", {})
    missing_cfg = config.get("missing_modality", {})
    fusion_cfg = config.get("fusion", {})
    policy = missing_cfg.get("policy", "independent")
    # Validate policy
    if policy not in ("skip", "zero_vector", "independent"):
        logger.warning(f"Unknown missing_modality policy '{policy}'. Falling back to 'independent'.")
        policy = "independent"

    pca_config = {
        "enabled": fusion_cfg.get("pca", {}).get("enabled", False),
        "image_components": fusion_cfg.get("pca", {}).get("image_components", 128),
        "audio_components": fusion_cfg.get("pca", {}).get("audio_components", 128),
        "normalize_per_modality": fusion_cfg.get("normalize_per_modality", True),
    }

    assembler = DatasetAssembler(
        metadata_path=metadata_path,
        missing_policy=policy,
        selection_config=selection_cfg,
        pca_config=pca_config,
    )
    
    # 3. Assemble and Fuse
    final_dataset = assembler.assemble(
        tabular_npz=tabular_npz,
        image_npz=image_npz,
        audio_npz=audio_npz
    )
    
    # 4. Export 
    if not final_dataset.empty:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Exporting to Parquet format...")
        final_dataset.to_parquet(out_parquet, index=False)
        
        # Also saving as CSV
        final_dataset.to_csv(out_csv, index=False)
        logger.info(f"Dataset successfully saved to {out_parquet} and {out_csv}")

        # Save fusion metadata so evaluate.py knows the actual dimensions
        fusion_meta = {
            "dim_tabular": assembler.dim_tabular_actual,
            "dim_image": assembler.dim_image_actual,
            "dim_audio": assembler.dim_audio_actual,
            "dim_total": len([c for c in final_dataset.columns if c.startswith("fused_feat_")]),
            "pca_enabled": pca_config.get("enabled", False),
            "bolasso_enabled": selection_cfg.get("enabled", False),
        }
        meta_path = os.path.join(output_dir, "fusion_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(fusion_meta, f, indent=2)
        logger.info(f"Fusion metadata saved to: {meta_path}")

        # --- EXPORT INDIVIDUAL MODALITY DATASETS ---
        # Find feature columns
        feat_cols = [c for c in final_dataset.columns if c.startswith("fused_feat_")]
        feat_cols.sort(key=lambda c: int(c.split("_")[-1]))
        
        total_feats = len(feat_cols)
        
        # Slices — use actual assembler dimensions (may differ from raw constants after PCA/Bolasso)
        d_tab = assembler.dim_tabular_actual
        d_img = assembler.dim_image_actual
        d_aud = assembler.dim_audio_actual
        
        if d_tab > 0 and 'has_tabular' in final_dataset.columns:
            tab_cols = feat_cols[:d_tab]
            tab_df = final_dataset[final_dataset['has_tabular'] == True][['sample_id', 'label'] + tab_cols].copy()
            tab_df = tab_df.rename(columns={col: f"tabular_feat_{i}" for i, col in enumerate(tab_cols)})
            
            out_tab_csv = os.path.join(output_dir, "tabular_final_dataset.csv")
            out_tab_parquet = os.path.join(output_dir, "tabular_final_dataset.parquet")
            tab_df.to_csv(out_tab_csv, index=False)
            tab_df.to_parquet(out_tab_parquet, index=False)
            logger.info(f"Saved independent tabular dataset to {out_tab_csv} ({len(tab_df)} samples)")
            
        if d_img > 0 and 'has_image' in final_dataset.columns:
            img_cols = feat_cols[d_tab : d_tab + d_img]
            img_df = final_dataset[final_dataset['has_image'] == True][['sample_id', 'label'] + img_cols].copy()
            img_df = img_df.rename(columns={col: f"image_feat_{i}" for i, col in enumerate(img_cols)})
            
            out_img_csv = os.path.join(output_dir, "image_final_dataset.csv")
            out_img_parquet = os.path.join(output_dir, "image_final_dataset.parquet")
            img_df.to_csv(out_img_csv, index=False)
            img_df.to_parquet(out_img_parquet, index=False)
            logger.info(f"Saved independent image dataset to {out_img_csv} ({len(img_df)} samples)")
            
        if d_aud > 0 and 'has_audio' in final_dataset.columns:
            aud_cols = feat_cols[d_tab + d_img :]
            aud_df = final_dataset[final_dataset['has_audio'] == True][['sample_id', 'label'] + aud_cols].copy()
            aud_df = aud_df.rename(columns={col: f"audio_feat_{i}" for i, col in enumerate(aud_cols)})
            
            out_aud_csv = os.path.join(output_dir, "audio_final_dataset.csv")
            out_aud_parquet = os.path.join(output_dir, "audio_final_dataset.parquet")
            aud_df.to_csv(out_aud_csv, index=False)
            aud_df.to_parquet(out_aud_parquet, index=False)
            logger.info(f"Saved independent audio dataset to {out_aud_csv} ({len(aud_df)} samples)")
    else:
        logger.error("Final dataset is empty. Check your sample_ids and metadata.")


if __name__ == "__main__":
    main()