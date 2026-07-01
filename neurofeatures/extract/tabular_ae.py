"""
NeuroFeature: Set-Based Tabular Dataset Autoencoder (Hybrid)

Handles PyTorch environments and gracefully falls back to scikit-learn
MLPRegressor-based Autoencoders if PyTorch DLL errors (such as Windows DLL loading bugs) occur.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Try importing torch; handle WinError 1114 DLL loading error
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except (ImportError, OSError) as e:
    logger.warning(f"PyTorch loading failed ({e}). Falling back to scikit-learn MLP Autoencoder.")
    HAS_TORCH = False
    DEVICE = None


# ---------------------------------------------------------------------------
# Descriptor Extractor (32 statistics per column)
# ---------------------------------------------------------------------------
def extract_column_descriptors(df: pd.DataFrame) -> np.ndarray:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return np.zeros((1, 32), dtype=np.float32)

    descriptors = []
    num_rows = len(df)

    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        if len(col_data) < 2:
            descriptors.append(np.zeros(32, dtype=np.float32))
            continue

        vals = col_data.values.astype(np.float64)
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        range_val = max_val - min_val
        median_val = float(np.median(vals))
        
        # Percentiles
        percentiles = [5, 10, 20, 25, 30, 40, 60, 70, 75, 80, 90, 95]
        ptiles_vals = np.percentile(vals, percentiles)
        
        iqr_val = ptiles_vals[8] - ptiles_vals[3]  # p75 - p25

        # Higher-order statistics
        try:
            skew_val = float(scipy_stats.skew(vals, bias=False))
        except Exception:
            skew_val = 0.0
        try:
            kurt_val = float(scipy_stats.kurtosis(vals, bias=False))
        except Exception:
            kurt_val = 0.0

        cov_val = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else 0.0

        # Distribution ratios
        zeros_ratio = float((vals == 0.0).sum() / num_rows)
        pos_ratio = float((vals > 0.0).sum() / num_rows)
        neg_ratio = float((vals < 0.0).sum() / num_rows)
        nan_ratio = float((num_rows - len(col_data)) / num_rows)

        # Unique counts
        unique_ratio = float(len(np.unique(vals)) / num_rows)

        # Variance & absolute deviations
        var_val = float(np.var(vals, ddof=1))
        sum_val = float(np.sum(vals))
        mad_val = float(np.mean(np.abs(vals - mean_val)))
        medad_val = float(np.median(np.abs(vals - median_val)))
        sem_val = std_val / np.sqrt(len(col_data))

        # Build final 32 stats
        stats = [
            mean_val, std_val, min_val, max_val, range_val,
            median_val, iqr_val, skew_val, kurt_val, cov_val,
            zeros_ratio, pos_ratio, neg_ratio, nan_ratio, unique_ratio,
            var_val, sum_val, mad_val, medad_val, sem_val
        ]
        # Append all 12 percentiles to reach 32 dimensions
        stats.extend(ptiles_vals.tolist())

        # Clean NaN/Inf
        stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        descriptors.append(stats)

    return np.array(descriptors, dtype=np.float32)


# ---------------------------------------------------------------------------
# PyTorch Model Definition
# ---------------------------------------------------------------------------
if HAS_TORCH:
    class TabularDatasetAE(nn.Module):
        def __init__(self, input_dim: int = 32, col_dim: int = 64, latent_dim: int = 768):
            super().__init__()
            self.col_encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, col_dim),
                nn.BatchNorm1d(col_dim),
                nn.ReLU()
            )
            self.latent_projection = nn.Sequential(
                nn.Linear(col_dim * 2, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
            self.col_decoder = nn.Sequential(
                nn.Linear(col_dim + col_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, input_dim)
            )

        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            col_embeddings = self.col_encoder(x)
            mean_pooled = torch.mean(col_embeddings, dim=0, keepdim=True)
            max_pooled = torch.max(col_embeddings, dim=0, keepdim=True)[0]
            pooled = torch.cat([mean_pooled, max_pooled], dim=1)
            latent_vector = self.latent_projection(pooled)
            return col_embeddings, latent_vector, pooled

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            col_embeddings, latent_vector, pooled = self.encode(x)
            C = col_embeddings.size(0)
            pooled_expanded = pooled.expand(C, -1)
            decoder_input = torch.cat([col_embeddings, pooled_expanded], dim=1)
            reconstruction = self.col_decoder(decoder_input)
            return reconstruction, latent_vector
else:
    class TabularDatasetAE:
        pass


# ---------------------------------------------------------------------------
# Training Helper - Scikit-Learn Fallback Model
# ---------------------------------------------------------------------------
class SklearnTabularAE:
    """Fallback Autoencoder utilizing scikit-learn's MLPRegressor."""
    def __init__(self, latent_dim: int = 128, out_dim: int = 768):
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        # MLP architecture: 768 input -> 128 bottleneck -> 768 reconstruction
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(latent_dim,),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        # Scale inputs
        X_scaled = self.scaler.fit_transform(X)
        logger.info(f"Training scikit-learn MLP Autoencoder on {X.shape[0]} samples...")
        self.mlp.fit(X_scaled, X_scaled)
        logger.info(f"Training completed. Final loss: {self.mlp.loss_:.6f}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode to bottleneck, and project back to 768-dim latent space."""
        X_scaled = self.scaler.transform(X)
        
        # Bottleneck hidden activations: X * W_0 + b_0
        w0 = self.mlp.coefs_[0]      # shape: (768, 128)
        b0 = self.mlp.intercepts_[0]  # shape: (128,)
        
        # ReLU activation
        latent = np.maximum(0, np.dot(X_scaled, w0) + b0) # shape: (N, 128)
        
        # Project bottleneck representation to 768 dimensions using second layer weights: latent * W_1 + b_1
        w1 = self.mlp.coefs_[1]      # shape: (128, 768)
        b1 = self.mlp.intercepts_[1]  # shape: (768,)
        
        latent_768 = np.dot(latent, w1) + b1 # shape: (N, 768)
        return latent_768.astype(np.float32)

    def save(self, filepath: str):
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump({
                "mlp": self.mlp,
                "scaler": self.scaler,
                "latent_dim": self.latent_dim,
                "out_dim": self.out_dim
            }, f)
        logger.info(f"Model saved to pickle: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SklearnTabularAE":
        import pickle
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        ae = cls(latent_dim=data["latent_dim"], out_dim=data["out_dim"])
        ae.mlp = data["mlp"]
        ae.scaler = data["scaler"]
        return ae


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def train_autoencoder(
    tabular_dir: str, 
    model_save_path: str,
    epochs: int = 150, 
    lr: float = 0.001
):
    # Determine fallback or torch
    if not HAS_TORCH:
        logger.info("Executing Scikit-Learn MLP fallback training...")
        # Since sklearn AE runs directly on the 768-dimensional statistical vectors,
        # we need to load tabular_features.npz
        npz_path = Path(model_save_path).parent / "tabular_features.npz"
        if not npz_path.exists():
            logger.error(f"Tabular features NPZ not found: {npz_path}. Please run tabular.py first.")
            return

        data = np.load(npz_path)
        features = data["features"] # shape: (208, 768)
        
        ae = SklearnTabularAE(latent_dim=128, out_dim=768)
        ae.fit(features)
        
        # Save as pickle
        pickle_path = str(Path(model_save_path).with_suffix(".pkl"))
        ae.save(pickle_path)
        return

    # Torch training pipeline
    logger.info(f"Scanning processed tabular CSV files under: {tabular_dir}")
    tab_path = Path(tabular_dir)
    if not tab_path.exists():
        logger.error(f"Tabular features folder not found: {tabular_dir}")
        return
        
    csv_files = list(tab_path.glob("*_features.csv"))
    if not csv_files:
        logger.error(f"No processed CSV feature files found in {tabular_dir}. Run tabular.py first.")
        return

    logger.info(f"Extracting column descriptors from {len(csv_files)} datasets...")
    dataset_matrices = []
    
    for file_p in csv_files:
        try:
            df = pd.read_csv(file_p)
            if "label" in df.columns:
                df = df.drop(columns=["label"])
            descriptors = extract_column_descriptors(df)
            dataset_matrices.append(descriptors)
        except Exception as e:
            logger.warning(f"Failed to extract descriptors for {file_p.name}: {e}")

    if not dataset_matrices:
        logger.error("No valid dataset descriptors extracted.")
        return

    all_descriptors = np.concatenate(dataset_matrices, axis=0)
    desc_mean = all_descriptors.mean(axis=0)
    desc_std = all_descriptors.std(axis=0) + 1e-8
    
    normalized_matrices = []
    for m in dataset_matrices:
        normalized_matrices.append((m - desc_mean) / desc_std)

    scale_dict = {"mean": desc_mean.tolist(), "std": desc_std.tolist()}
    scale_path = Path(model_save_path).parent / "tabular_ae_scale.json"
    with open(scale_path, "w") as f:
        json.dump(scale_dict, f, indent=2)

    # PyTorch Dataset
    class ColDescriptorDataset(Dataset):
        def __init__(self, data_list): self.data_list = data_list
        def __len__(self): return len(self.data_list)
        def __getitem__(self, idx): return torch.tensor(self.data_list[idx], dtype=torch.float32)

    loader = DataLoader(
        ColDescriptorDataset(normalized_matrices), 
        batch_size=1, 
        shuffle=True, 
        collate_fn=lambda x: x
    )

    model = TabularDatasetAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    logger.info("Training PyTorch Autoencoder...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            reconstructed, _ = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 15 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] | Reconstruction MSE Loss: {epoch_loss/len(loader):.6f}")

    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Saved PyTorch checkpoint: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or inspect Tabular Autoencoder")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train", action="store_true", help="Train the autoencoder")
    args = parser.parse_args()
    
    config = {}
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            import yaml
            config = yaml.safe_load(f)
            
    paths = config.get("paths", {})
    output_dir = paths.get("output_dir", "neurofeatures/outputs")
    tab_csv_dir = paths.get("tabular_csv_dir", "neurofeatures/outputs/tabular")
    
    model_path = os.path.join(output_dir, "tabular_ae.pth")
    
    if args.train:
        train_autoencoder(
            tabular_dir=tab_csv_dir,
            model_save_path=model_path,
            epochs=120,
            lr=0.001
        )
    else:
        if HAS_TORCH:
            model = TabularDatasetAE()
            print(model)
        else:
            print("PyTorch not loaded. Falling back to scikit-learn autoencoder.")
