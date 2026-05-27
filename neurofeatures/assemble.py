"""
NeuroFeature Pipeline: Dataset Assembly and Baseline Fusion

This module handles Stage 3 and 4 of the NeuroFeature pipeline.
It merges extracted modality features (Text, Image, Audio) based on `sample_id`,
applies the configured missing-modality policies, and performs baseline 
concatenation fusion to export a model-ready tabular dataset.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Define expected embedding dimensions based on our extractors
DIM_TEXT = 768   # DistilBERT
DIM_IMAGE = 1280 # MobileNetV2
DIM_AUDIO = 768  # Wav2Vec2 Base


class DatasetAssembler:
    def __init__(
        self, 
        metadata_path: str,
        missing_policy: Literal['skip', 'zero_vector'] = 'zero_vector'
    ):
        """
        Args:
            metadata_path: Path to the master CSV containing sample_ids and labels.
            missing_policy: How to handle samples missing a modality.
        """
        self.metadata_path = Path(metadata_path)
        self.missing_policy = missing_policy
        
        logger.info(f"Initializing Assembler | Policy: {self.missing_policy.upper()}")
        self.metadata_df = pd.read_csv(self.metadata_path)
        
        # Ensure sample_id is a string for safe merging
        self.metadata_df['sample_id'] = self.metadata_df['sample_id'].astype(str)

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

    def _load_csv_features(self, csv_path: Path) -> Dict[str, np.ndarray]:
        """Loads text features from CSV and maps filename to feature array."""
        if not csv_path.exists():
            logger.warning(f"Feature file not found: {csv_path}")
            return {}
            
        df = pd.read_csv(csv_path)
        feature_dict = {}
        
        # Find the bert_emb columns dynamically
        emb_cols = [c for c in df.columns if c.startswith('bert_emb_')]
        
        for _, row in df.iterrows():
            base_name = Path(row['filename']).stem
            feat_vec = row[emb_cols].values.astype(np.float32)
            feature_dict[base_name] = feat_vec
            
        logger.info(f"Loaded {len(feature_dict)} features from {csv_path.name}")
        return feature_dict

    def assemble(self, text_csv: str, image_npz: str, audio_npz: str) -> pd.DataFrame:
        """
        Merges all modalities based on the master metadata file.
        """
        logger.info("Loading extracted modality features...")
        text_feats = self._load_csv_features(Path(text_csv))
        image_feats = self._load_npz_features(Path(image_npz), DIM_IMAGE)
        audio_feats = self._load_npz_features(Path(audio_npz), DIM_AUDIO)
        
        assembled_rows = []
        skipped_count = 0
        
        logger.info("Assembling multimodal dataset...")
        
        for _, row in self.metadata_df.iterrows():
            sample_id = str(row['sample_id'])
            
            # 1. Fetch features (returns array or None)
            t_vec = text_feats.get(sample_id)
            i_vec = image_feats.get(sample_id)
            a_vec = audio_feats.get(sample_id)
            
            # 2. Check for missing modalities and apply 'skip' policy
            if t_vec is None or i_vec is None or a_vec is None:
                if self.missing_policy == 'skip':
                    skipped_count += 1
                    continue

            # 3. Resolve strict typing for Pylance with explicit annotations
            t_arr: np.ndarray = t_vec if t_vec is not None else np.zeros(DIM_TEXT, dtype=np.float32)
            i_arr: np.ndarray = i_vec if i_vec is not None else np.zeros(DIM_IMAGE, dtype=np.float32)
            a_arr: np.ndarray = a_vec if a_vec is not None else np.zeros(DIM_AUDIO, dtype=np.float32)

            # 4. Baseline Fusion (Concatenation)
            # Using a tuple () completely satisfies Pylance's overload requirements
            fused_vector = np.concatenate((t_arr, i_arr, a_arr))
            
            # 5. Build the row dictionary
            row_data = {
                'sample_id': sample_id,
                'label': row.get('label', np.nan) 
            }
            
            # Add the 2816 concatenated features dynamically
            for idx, val in enumerate(fused_vector):
                row_data[f"fused_feat_{idx}"] = val
                
            assembled_rows.append(row_data)

        if self.missing_policy == 'skip':
            logger.info(f"Skipped {skipped_count} samples due to missing modalities.")
            
        final_df = pd.DataFrame(assembled_rows)
        logger.info(f"Assembly complete! Final Dataset Shape: {final_df.shape}")
        return final_df


def main():
    """CLI entry point for the NeuroFeature Dataset Assembler."""
    
    # 1. Setup Paths based on your architecture
    current_dir = Path(__file__).resolve().parent
    base_dir = current_dir.parent / "data"
    
    # Ensure this file exists based on your README specifications
    master_metadata = current_dir / "metadata_example.csv" 
    
    out_parquet = base_dir / "neurofeature_final_dataset.parquet"
    out_csv = base_dir / "neurofeature_final_dataset.csv"
    
    if not master_metadata.exists():
        logger.error(f"Cannot assemble dataset. Missing master metadata at {master_metadata}")
        return

    # 2. Initialize Assembler
    assembler = DatasetAssembler(
        metadata_path=str(master_metadata),
        missing_policy='zero_vector' 
    )
    
    # 3. Assemble and Fuse
    final_dataset = assembler.assemble(
        text_csv=str(base_dir / "metadata_text.csv"),
        image_npz=str(base_dir / "image_features.npz"),
        audio_npz=str(base_dir / "audio_features.npz")
    )
    
    # 4. Export 
    if not final_dataset.empty:
        logger.info(f"Exporting to Parquet format (Highly Recommended for ML)...")
        final_dataset.to_parquet(out_parquet, index=False)
        
        # Also saving as CSV just in case you want to view it manually
        final_dataset.to_csv(out_csv, index=False)
        logger.info(f"Dataset successfully saved to {out_parquet} and {out_csv}")
    else:
        logger.error("Final dataset is empty. Check your sample_ids and metadata.")


if __name__ == "__main__":
    main()