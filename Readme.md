# NeuroFeature – Automated Multimodal Feature Engineering Pipeline

NeuroFeature is an end-to-end feature engineering pipeline that converts raw **text**, **audio**, and **image** data into clean, fixed-size, model-ready datasets. It is designed to reduce manual feature engineering time for data scientists and ML engineers by standardizing preprocessing, feature extraction, feature selection, and dataset assembly.

NeuroFeature does **not** train the final ML model. Instead, it focuses on the data side of the workflow and produces high-quality feature tables that can be used with any downstream ML model or training framework.

---

## ✨ Features

- **Automated multimodal ingestion**
  - Load datasets via a metadata file (CSV/JSON) that maps each `sample_id` to its text, image, and audio files.
  - Validate file paths and build a unified index of samples and available modalities.

- **Modality-specific feature extraction**
  - **Text**: Transformer-based embeddings (e.g., BERT-style sentence or document vectors).
  - **Images**: CNN-based embeddings (e.g., MobileNet-style feature vectors).
  - **Audio (primary)**: Pretrained audio embeddings for speech/music tasks.
  - **Audio (fallback)**: librosa-based handcrafted features (MFCCs, spectral and temporal statistics) for lightweight or low-resource setups.

- **Clean dataset assembly**
  - Join per-modality feature blocks on `sample_id` to create a single tabular dataset.
  - Apply normalization/scaling, simple outlier handling, and duplicate detection as configured.

- **Stability-oriented feature selection (optional)**
  - Bootstrap-resampled Lasso used **only** on handcrafted numerical features such as audio statistics, metadata, and other structured descriptors.
  - Dense pretrained embeddings (BERT, CNN, audio encoders) are **not** passed through this selector to avoid misusing linear sparsity methods on high-dimensional, correlated embeddings.

- **Experimental multimodal fusion**
  - **Baseline**: Simple concatenation of per-modality feature blocks.
  - **Experimental**: Gated or attention-style fusion layers for supervised tasks, requiring a target label and providing learned modality weighting on top of extracted features. Uses additive (weighted-sum) fusion rather than multiplicative gating to avoid modality suppression when any single modality produces a near-zero weight.

- **Explicit missing-modality handling**
  - Configurable policies for missing text/image/audio at sample level: skip the modality, insert a mask/zero vector, or use fallback features.
  - No silent corruption or hidden assumptions about incomplete samples.

- **Streamlit web interface**
  - Upload metadata and data archives.
  - Configure which modalities and feature types to use.
  - Monitor preprocessing progress and inspect basic stats.
  - Download the final model-ready dataset.

- **Model-agnostic outputs**
  - Export clean CSV/Parquet with one row per sample and numeric feature columns plus optional label columns.
  - Ready for scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow, or custom pipelines.

---

## 🧠 Project Scope

NeuroFeature is a **feature engineering and dataset construction tool** for multimodal ML.

It **is**:
- A way to automate preprocessing for text, audio, and images.
- A way to produce consistent, model-ready tabular datasets.
- A way to experiment with different feature sets and fusion strategies.

It is **not**:
- A guarantee of better accuracy on every task.
- A replacement for task-specific modeling and evaluation.
- A full end-to-end prediction system.

---

## 🏗️ Architecture

NeuroFeature follows a four-stage architecture:

### 1. Ingest
- Input: metadata file (CSV/JSON) with at least:
  - `sample_id` (string/int)
  - `label` (optional, for supervised tasks)
  - `text_path`, `image_path`, `audio_path` (optional per-row, can be missing)
- Steps:
  - Validate that referenced files exist.
  - Build a unified table of samples and available modalities.
  - Record missing-modality information for each sample.

### 2. Extract

- **Text pipeline**: Clean → tokenize → embed via pretrained transformer → fixed-size vector per sample.
- **Image pipeline**: Load + resize + normalize → pretrained CNN backbone → penultimate-layer embedding.
- **Audio pipeline**:
  - *Primary*: Pretrained audio encoder (e.g., wav2vec-style) → fixed-size semantic embedding.
  - *Fallback*: librosa MFCCs + spectral/temporal statistics → aggregated handcrafted vector.

### 3. Assemble & Select
- Merge per-modality blocks on `sample_id`.
- Apply scaling, outlier clipping, and integrity checks.
- Optionally run bootstrap-resampled Lasso **on handcrafted features only** to retain consistently selected dimensions.

### 4. Fuse & Export
- **Concatenation baseline**: column-wise join of all modality blocks.
- **Experimental learned fusion**: additive attention over modality blocks given a supervised target.
- Export as CSV or Parquet with schema metadata (feature names, modality origin, scaling parameters).

---

## 🔍 Missing Modality Handling

| Policy | Behaviour |
|---|---|
| `skip` | Omit that modality's features for the affected sample; rely on remaining modalities. |
| `zero_vector` | Insert a fixed zero vector of the expected size for that modality. |
| `fallback_audio` | If pretrained audio encoder fails, fall back to librosa handcrafted features when audio file is present. |

The active policy is recorded in the run config and logs for full reproducibility.

---

## 🛠️ Tech Stack

| Layer | Libraries |
|---|---|
| Core | Python, NumPy, Pandas |
| Feature selection | scikit-learn (Lasso, scalers) |
| Deep learning | PyTorch, Hugging Face transformers |
| Audio | librosa, soundfile |
| Frontend | Streamlit |
| Export | pyarrow (Parquet), CSV |

---

## 📦 Installation

```bash
git clone https://github.com/DevShambhvi/NeuroFeature.git
cd NeuroFeature
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate.bat   # Windows
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Prepare your metadata file

```text
sample_id,label,text_path,image_path,audio_path
1,positive,data/text/1.txt,data/images/1.jpg,data/audio/1.wav
2,negative,data/text/2.txt,,data/audio/2.wav
3,neutral,,,data/audio/3.wav
```

Any path column can be empty — NeuroFeature handles missing modalities per the configured policy.

### 2. Configure the pipeline

Edit `config.yaml` to select modalities, feature types, selection, and fusion options.

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Then in the browser:
1. Upload your metadata file and zipped data folder.
2. Choose modalities, feature types, and optional steps.
3. Start the pipeline and monitor progress.
4. Download the final feature dataset.

---

## 🎯 Use Cases

- Rapid dataset creation for multimodal experiments.
- Building feature sets for classical ML models from text/audio/image inputs.
- Comparing different feature extraction and fusion strategies on the same data.
- Educational projects on multimodal ML and feature engineering.

---

## 📌 Design Principles

- **Model-agnostic**: Prepare data, don't dictate the model.
- **Transparent**: Make selection, fusion, and missing-modality behaviour explicit.
- **Pragmatic**: Use standard, well-supported libraries and patterns.
- **Honest**: No exaggerated claims of universal stability or accuracy gains.

---

## 🤝 Contributing

Contributions are welcome. Some ideas:
- New pretrained encoders for specific domains (medical images, domain-specific text).
- Additional handcrafted feature sets.
- Improved missing-modality strategies and evaluation utilities.
- Better logging and configuration management.

Open an issue or submit a pull request.

---

## 📄 License

MIT — see `LICENSE` for details.