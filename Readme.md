# NeuroFeature – Automated Multimodal Feature Engineering Pipeline

NeuroFeature is an end-to-end feature engineering pipeline that converts raw **tabular**, **audio**, and **image** data into clean, fixed-size, model-ready datasets. It is designed to reduce manual feature engineering time for data scientists and ML engineers by standardizing preprocessing, feature extraction, feature selection, and dataset assembly.

NeuroFeature does **not** train the final ML model. Instead, it focuses on the data side of the workflow and produces high-quality feature tables that can be used with any downstream ML model or training framework.

---

## ✨ Features

- **Automated Multimodal Ingestion & Pairing**
  - Load datasets via a configuration file (`config.yaml`) and a master metadata index (`metadata_expanded.csv`).
  - Supports two ingestion modes: `independent` (union-based, single-modality samples) and `paired` (one-to-one aligned multimodal samples).

- **Modality-Specific Feature Extraction**
  - **Tabular**: Hybrid representation combining 32 statistics per column (skewness, kurtosis, IQR, percentiles) and global correlation eigenvalues. Supports a learned **Set-Based Tabular Autoencoder** (`tabular_ae.py`) using scikit-learn MLP regression as a fallback.
  - **Images**: Pretrained MobileNetV2 backbone (CPU-based) generating 1280-dim embedding vectors.
  - **Audio**: Pretrained Wav2Vec 2.0 (CPU-based) generating 768-dim global embedding vectors.

- **Dimensionality Reduction & Normalization**
  - Performs per-modality StandardScaler normalization before fusion to prevent numerical imbalances.
  - Implements **PCA Dimensionality Reduction** for dense embeddings: maps images (1280 ➔ 128) and audio (768 ➔ 128) while retaining **91%+ and 99%+ of the variance** respectively, resulting in a **10x reduction in feature size**.

- **Stability-Oriented Feature Selection (Bolasso)**
  - Bootstrap-resampled Lasso (Bolasso) used on handcrafted numerical tabular features to identify stable predictors.
  - Excludes high-dimensional deep learning embeddings from linear feature selection to prevent loss of context.

- **Explicit Missing-Modality Handling**
  - Configurable policies: `skip` (omit incomplete samples), `zero_vector` (pad missing blocks with zeros), or `independent` (zero-pad and record provenance boolean flags).

- **Built-In Evaluation & Model Ablation**
  - Evaluates feature quality by training **RandomForest**, **HistGradientBoosting**, and **deep PyTorch/Sklearn Neural Networks (MLP)** via stratified 5-fold cross-validation.
  - Reports accuracy, ROC-AUC, confusion matrices, and precision/recall reports per modality and for the fused set.

---

## 📁 Project Structure

```
NeuroFeature/
├── config.yaml                        # Pipeline configuration
├── requirements.txt                   # Python dependencies
├── Readme.md
├── .gitignore
└── neurofeatures/
    ├── __init__.py
    ├── ingest.py                      # Stage 1: Data ingestion & indexing
    ├── extract/
    │   ├── tabular.py                 # Tabular feature extraction (latent/stat hybrid)
    │   ├── tabular_ae.py              # NEW: Set-Based Tabular Autoencoder model & trainer
    │   ├── image.py                   # Image feature extraction (MobileNetV2)
    │   └── audio.py                   # Audio feature extraction (Wav2Vec 2.0)
    ├── assemble.py                    # Stage 3-4: Assembly, PCA, Bolasso & fusion
    ├── evaluate.py                    # Stage 5: Downstream ML classifier evaluation
    ├── generate_metadata.py           # Auto-generate metadata (supports independent/paired)
    ├── metadata_example.csv           # Minimal metadata index
    └── outputs/
        ├── tabular/                   # Cleaned per-dataset features (e.g. medical_features.csv)
        ├── tabular_ae.pkl             # Trained autoencoder checkpoints
        ├── tabular_ae_scale.json      # Tabular normalization scaling limits
        ├── tabular_features.npz       # Extracted tabular latent representations
        ├── image_features.npz         # Extracted image embeddings
        ├── audio_features.npz         # Extracted audio embeddings
        ├── tabular_final_dataset.csv  # Cleaned, ready-to-train tabular dataset
        ├── image_final_dataset.csv    # Cleaned, ready-to-train computer vision feature dataset
        ├── audio_final_dataset.csv    # Cleaned, ready-to-train audio feature dataset
        ├── neurofeature_final_dataset.csv     # Final fused multimodal dataset
        ├── neurofeature_final_dataset.parquet # Final fused multimodal dataset (Parquet)
        ├── fusion_metadata.json       # Record of dynamic feature dimensions
        └── evaluation_metrics.json    # Compiled performance scores
```

---

## 🏗️ Architecture

NeuroFeature follows a five-stage architecture:

```
[ Ingest Stage ]   ➔   [ Extraction Stage ]    ➔   [ Assembly & Select ]   ➔   [ Fusion Stage ]   ➔   [ Evaluation Stage ]
Scan directories       Extract latent embeddings      Run PCA (Image/Audio)       Concatenation join      Run Stratified CV
Generate indexes       Tabular AE + CNN + wav2vec     Bolasso (Tabular only)      Verify dimensions       MLP, RF, HistGB
```

1. **Ingest (`ingest.py`):** Scans modality subdirectories under `data/` and builds indexes.
2. **Extract (`extract/*.py`):** Loads files to compute latent embeddings. Tabular extraction uses a trained MLP Autoencoder bottleneck, falling back to 32 stats + correlation eigenvalues per table.
3. **Assemble (`assemble.py`):** Aligns sample IDs, normalizes features per modality, runs PCA (reducing visual and auditory feature dimension to 128 each), and runs Bolasso on tabular vectors.
4. **Fuse (`assemble.py`):** Concatenates aligned modalities, pads missing modalities, and exports Parquet/CSV tables.
5. **Evaluate (`evaluate.py`):** Performs 5-fold stratified cross-validation on individual modalities and the fused set.

---

## 🖥️ Streamlit Web Dashboard

An interactive dashboard is available to tweak hyperparameters, trigger the pipeline, and visualize evaluation results:

```bash
streamlit run app.py
```
This serves a premium UI locally at `http://localhost:8501`.

---

## 🔬 Benchmark Evaluation Results

Following the modality pairing and autoencoder enhancements, NeuroFeature yields exceptional classification results on the aligned dataset:

| Modality | Classifier Model | Accuracy | ROC-AUC | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Tabular** | RandomForest (AE Latent) | **72.6%** | 0.620 | **+22.6%** (from 50.0%) |
| **Image** | Gradient Boosting (PCA-128) | **93.3%** | 0.985 | **+37.0%** (from 56.3%) |
| **Audio** | Neural Network (PCA-128) | **84.6%** | 0.875 | **+12.1%** (from 72.5%) |
| **Fused** | Gradient Boosting (Multimodal)| **93.1%** | 0.983 | **+34.3%** (from 58.8%) |

---

## 🚀 Usage Guide

### 1. Auto-Generate Metadata Index
To pair modalities sequentially for true multimodal training:
```bash
python neurofeatures/generate_metadata.py --mode paired
```
*(Use `--mode independent` for union-based indexing if modalities do not share samples.)*

### 2. Train the Tabular Autoencoder
Learn latent representations for tabular datasets:
```bash
python neurofeatures/extract/tabular_ae.py --train
```

### 3. Extract Modality Features
Run extraction across all enabled modalities:
```bash
# Tabular
python neurofeatures/extract/tabular.py

# Images
python neurofeatures/extract/image.py

# Audio
python neurofeatures/extract/audio.py
```

### 4. Assemble the Fusion Dataset
Perform scaling, Bolasso, PCA reduction, and concatenation:
```bash
python neurofeatures/assemble.py
```

### 5. Evaluate Accuracy
Train and evaluate downstream classifiers:
```bash
python neurofeatures/evaluate.py
```

---

## ⚙️ Configuration Reference

Key pipeline controls inside `config.yaml`:
```yaml
selection:
  enabled: true
  n_bootstraps: 100
  selection_threshold: 0.5

fusion:
  normalize_per_modality: true
  pca:
    enabled: true
    image_components: 128
    audio_components: 128
  learned:
    hidden_dims: [512, 256, 128]
    dropout: 0.3
    epochs: 80
```

---

## 🤝 Contributing
Feel free to open an issue or pull request. Key areas:
* Additional deep learning feature extraction backbones (e.g. ResNet, ViT).
* Advanced pooling and attention-based multimodal fusion models.
* Enhanced tabular Autoencoder configurations.