"""
NeuroFeature – Interactive Multimodal Pipeline Dashboard

Provides a premium dark-mode interface to:
1. Adjust hyperparameters and sync directly with config.yaml.
2. Execute all pipeline steps sequentially with real-time log output.
3. Explore dataset structures and download final model-ready sets.
4. Visualize model accuracies and evaluation metrics using Plotly.
"""

import os
import sys
import yaml
import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set Page Config
st.set_page_config(
    page_title="NeuroFeature Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS for Sleek Dark Theme (Glassmorphism & Glowing accents)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main Background and Colors */
    .reportview-container {
        background: #0E1117;
    }
    
    /* Transparent glassmorphism cards */
    div.element-container {
        background-color: transparent;
    }
    
    .glass-card {
        background: rgba(33, 38, 47, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    
    /* Glowing status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .status-active {
        background-color: rgba(31, 111, 235, 0.2);
        color: #58a6ff;
        border: 1px solid rgba(31, 111, 235, 0.4);
    }
    .status-success {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.3);
    }
    
    /* Gradient headers */
    .gradient-text {
        background: linear-gradient(120deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Styled Subprocess Logs Console */
    .console-log {
        background-color: #0d1117;
        font-family: 'Courier New', Courier, monospace;
        color: #8b949e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #21262d;
        height: 250px;
        overflow-y: auto;
        font-size: 0.9em;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Config Management Helpers
# ---------------------------------------------------------------------------
CONFIG_PATH = "config.yaml"

def load_config() -> dict:
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}

def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Pipeline Subprocess Runner
# ---------------------------------------------------------------------------
def run_pipeline_step(command_args: list, status_text: str):
    """Executes a python step in subprocess and yields output lines for logging."""
    process = subprocess.Popen(
        [sys.executable] + command_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Yield output log line-by-line
    for line in iter(process.stdout.readline, ""):
        yield line
    process.stdout.close()
    
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command_args)


# ---------------------------------------------------------------------------
# Load Current Data State
# ---------------------------------------------------------------------------
config = load_config()
output_dir = config.get("paths", {}).get("output_dir", "neurofeatures/outputs")

# Resolve datasets
final_dataset_csv = os.path.join(output_dir, "neurofeature_final_dataset.csv")
final_dataset_parquet = os.path.join(output_dir, "neurofeature_final_dataset.parquet")
metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
fusion_meta_file = os.path.join(output_dir, "fusion_metadata.json")

# Header Title
st.markdown("<h1 class='gradient-text'>🧠 NeuroFeature Control Center</h1>", unsafe_allow_html=True)
st.markdown("Automated Multimodal Feature Engineering & Alignment Dashboard")
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar Panel: Config Adjustments
# ---------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Pipeline Configuration")

# Ingestion Modalities
st.sidebar.markdown("**Modalities Active**")
tab_active = st.sidebar.checkbox("Tabular", value=config.get("modalities", {}).get("tabular", True))
img_active = st.sidebar.checkbox("Image", value=config.get("modalities", {}).get("image", True))
aud_active = st.sidebar.checkbox("Audio", value=config.get("modalities", {}).get("audio", True))

config["modalities"] = {
    "tabular": tab_active,
    "image": img_active,
    "audio": aud_active
}

# Metadata settings
st.sidebar.markdown("---")
st.sidebar.markdown("**Ingestion Mode**")
pairing_mode = st.sidebar.selectbox(
    "Sample pairing mode",
    options=["paired", "independent"],
    index=0 if config.get("missing_modality", {}).get("policy") == "independent" else 1,
    help="paired: aligns files to 228 multimodal samples. independent: scans disjoint sets."
)
config["missing_modality"] = {
    "policy": "independent" if pairing_mode == "paired" else "zero_vector"
}

# PCA Settings
st.sidebar.markdown("---")
st.sidebar.markdown("**PCA Feature Reduction**")
pca_enabled = st.sidebar.toggle("PCA Enabled", value=config.get("fusion", {}).get("pca", {}).get("enabled", True))
img_pca = st.sidebar.slider("Visual Dimensions", 16, 512, int(config.get("fusion", {}).get("pca", {}).get("image_components", 128)), step=16)
aud_pca = st.sidebar.slider("Auditory Dimensions", 16, 512, int(config.get("fusion", {}).get("pca", {}).get("audio_components", 128)), step=16)

config["fusion"] = {
    "method": "concatenation",
    "normalize_per_modality": True,
    "pca": {
        "enabled": pca_enabled,
        "image_components": img_pca,
        "audio_components": aud_pca
    },
    "learned": config.get("fusion", {}).get("learned", {
        "hidden_dims": [512, 256, 128],
        "dropout": 0.3,
        "epochs": 80,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64
    })
}

# Bolasso Feature Selection
st.sidebar.markdown("---")
st.sidebar.markdown("**Feature Selection (Bolasso)**")
bolasso_enabled = st.sidebar.toggle("Bolasso Enabled", value=config.get("selection", {}).get("enabled", True))
lasso_alpha = st.sidebar.slider("L1 Regularization Alpha", 0.001, 0.1, float(config.get("selection", {}).get("lasso_alpha", 0.01)), step=0.002, format="%.3f")
selection_thresh = st.sidebar.slider("Stability Threshold", 0.3, 0.9, float(config.get("selection", {}).get("selection_threshold", 0.5)), step=0.05)

config["selection"] = {
    "enabled": bolasso_enabled,
    "apply_to": "handcrafted_only",
    "n_bootstraps": 100,
    "lasso_alpha": lasso_alpha,
    "selection_threshold": selection_thresh
}

# Neural Network settings
st.sidebar.markdown("---")
st.sidebar.markdown("**MLP Neural Network Classifier**")
nn_epochs = st.sidebar.slider("Training Epochs", 10, 200, int(config.get("fusion", {}).get("learned", {}).get("epochs", 80)), step=10)
nn_lr = st.sidebar.selectbox("Learning Rate", [0.01, 0.005, 0.001, 0.0005, 0.0001], index=2)

config["fusion"]["learned"]["epochs"] = nn_epochs
config["fusion"]["learned"]["learning_rate"] = nn_lr

# Save back to config.yaml if changed
save_config(config)


# ---------------------------------------------------------------------------
# Main Layout Tabs
# ---------------------------------------------------------------------------
tab_run, tab_viz, tab_explore = st.tabs(["🚀 Run Pipeline", "📊 Evaluation Metrics", "📁 Dataset Explorer"])

# ---------------------------------------------------------------------------
# TAB 1: RUN PIPELINE
# ---------------------------------------------------------------------------
with tab_run:
    st.markdown("### 📥 Raw Data Ingestion Dock")
    st.markdown("Upload raw files to their respective modality directories. Once files are uploaded, run Step 1 to auto-generate sample indexes.")
    
    # Resolve directories
    paths_cfg = config.get("paths", {})
    t_dir = paths_cfg.get("tabular_input_dir", "neurofeatures/data/tabular")
    i_dir = paths_cfg.get("image_dir", "neurofeatures/data/images")
    a_dir = paths_cfg.get("audio_dir", "neurofeatures/data/audio")
    
    col_up_tab, col_up_img, col_up_aud = st.columns(3)
    
    with col_up_tab:
        uploaded_tab = st.file_uploader("Upload Tabular CSVs (or .zip)", type=["csv", "zip"], accept_multiple_files=True, key="up_tabular")
        if uploaded_tab:
            import zipfile
            import io
            count_tab = 0
            for f in uploaded_tab:
                if f.name.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(f.read())) as z:
                        for zinfo in z.infolist():
                            if zinfo.is_dir():
                                continue
                            filename = os.path.basename(zinfo.filename)
                            if filename.lower().endswith(".csv") and not filename.startswith("."):
                                out_path = os.path.join(t_dir, filename)
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                with open(out_path, "wb") as out_f:
                                    out_f.write(z.read(zinfo.filename))
                                count_tab += 1
                else:
                    out_path = os.path.join(t_dir, f.name)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "wb") as out_f:
                        out_f.write(f.getbuffer())
                    count_tab += 1
            st.success(f"Successfully saved {count_tab} CSV dataset(s)!")
            
    with col_up_img:
        uploaded_img = st.file_uploader("Upload Images (or .zip)", type=["png", "jpg", "jpeg", "zip"], accept_multiple_files=True, key="up_images")
        if uploaded_img:
            import zipfile
            import io
            count_img = 0
            for f in uploaded_img:
                if f.name.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(f.read())) as z:
                        for zinfo in z.infolist():
                            if zinfo.is_dir():
                                continue
                            filename = os.path.basename(zinfo.filename)
                            if filename.lower().endswith((".png", ".jpg", ".jpeg")) and not filename.startswith("."):
                                out_path = os.path.join(i_dir, filename)
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                with open(out_path, "wb") as out_f:
                                    out_f.write(z.read(zinfo.filename))
                                count_img += 1
                else:
                    out_path = os.path.join(i_dir, f.name)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "wb") as out_f:
                        out_f.write(f.getbuffer())
                    count_img += 1
            st.success(f"Successfully saved {count_img} image file(s)!")
            
    with col_up_aud:
        uploaded_aud = st.file_uploader("Upload Audios (or .zip)", type=["wav", "zip"], accept_multiple_files=True, key="up_audio")
        if uploaded_aud:
            import zipfile
            import io
            count_aud = 0
            for f in uploaded_aud:
                if f.name.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(f.read())) as z:
                        for zinfo in z.infolist():
                            if zinfo.is_dir():
                                continue
                            filename = os.path.basename(zinfo.filename)
                            if filename.lower().endswith(".wav") and not filename.startswith("."):
                                out_path = os.path.join(a_dir, filename)
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                with open(out_path, "wb") as out_f:
                                    out_f.write(z.read(zinfo.filename))
                                count_aud += 1
                else:
                    out_path = os.path.join(a_dir, f.name)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "wb") as out_f:
                        out_f.write(f.getbuffer())
                    count_aud += 1
            st.success(f"Successfully saved {count_aud} audio file(s)!")
            
    # Count existing files in each folder
    c_tab = len(list(Path(t_dir).glob("*.csv"))) if os.path.exists(t_dir) else 0
    c_img = len(list(Path(i_dir).glob("*"))) if os.path.exists(i_dir) else 0
    c_aud = len(list(Path(a_dir).glob("*.wav"))) if os.path.exists(a_dir) else 0
    
    st.markdown(
        f"<div style='font-size: 0.9em; color: #8b949e; padding: 5px; margin-bottom: 10px;'>"
        f"📂 Current folder status: <b>{c_tab}</b> Tabular CSVs | <b>{c_img}</b> Images | <b>{c_aud}</b> Audio files (.wav)"
        f"</div>",
        unsafe_allow_html=True
    )
    
    clear_col, _ = st.columns([1, 4])
    with clear_col:
        clear_data = st.button("🗑️ Clear Raw Data folders", help="Deletes all raw files inside tabular, image, and audio directories.")
        if clear_data:
            import shutil
            for folder in [t_dir, i_dir, a_dir]:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        file_path = os.path.join(folder, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            st.error(f"Failed to delete {file_path}: {e}")
            st.toast("Raw directories cleared!", icon="🗑️")
            st.rerun()
            
    st.markdown("---")
    st.markdown("### 🎛️ Execution Controls")
    st.markdown("Trigger pipeline steps individually or execute sequentially. Command line outputs will log dynamically below.")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        run_meta = st.button("Step 1: Generate Metadata", use_container_width=True)
    with col2:
        run_ae = st.button("Step 2: Train Autoencoder", use_container_width=True)
    with col3:
        run_extract = st.button("Step 3: Extract Features", use_container_width=True)
    with col4:
        run_assemble = st.button("Step 4: Fuse & Assemble", use_container_width=True)
    with col5:
        run_eval = st.button("Step 5: Run Evaluation", use_container_width=True)
        
    st.markdown("---")
    
    # Progress & Logging Docker
    log_area = st.empty()
    progress_bar = st.empty()
    
    # Execution Logic
    executed_tasks = []
    
    if run_meta:
        executed_tasks.append((["neurofeatures/generate_metadata.py", "--mode", pairing_mode], "Generating dataset indexes and mapping paired sample boundaries"))
    elif run_ae:
        executed_tasks.append((["neurofeatures/extract/tabular_ae.py", "--train"], "Training tabular dataset autoencoder bottleneck model"))
    elif run_extract:
        if tab_active:
            executed_tasks.append((["neurofeatures/extract/tabular.py"], "Extracting tabular representations from CSV tables"))
        if img_active:
            executed_tasks.append((["neurofeatures/extract/image.py"], "Extracting visual embeddings from image files"))
        if aud_active:
            executed_tasks.append((["neurofeatures/extract/audio.py"], "Extracting auditory embeddings from WAV files"))
    elif run_assemble:
        executed_tasks.append((["neurofeatures/assemble.py"], "Joining aligned modalities, normalizing, and running PCA reductions"))
    elif run_eval:
        executed_tasks.append((["neurofeatures/evaluate.py"], "Evaluating features across 4 modality sets using CV"))

    if executed_tasks:
        progress_bar.progress(10)
        logs = []
        
        total_tasks = len(executed_tasks)
        for idx, (command_args, task_desc) in enumerate(executed_tasks):
            with st.status(f"⏳ Processing [{idx+1}/{total_tasks}]: {task_desc}...", expanded=True) as status_box:
                try:
                    for log_line in run_pipeline_step(command_args, task_desc):
                        logs.append(log_line)
                        # Show last 12 lines of console output
                        log_area.markdown(
                            f"<div class='console-log'>{''.join(logs[-12:])}</div>", 
                            unsafe_allow_html=True
                        )
                    status_box.update(label=f"✅ Completed [{idx+1}/{total_tasks}]: {task_desc}", state="complete")
                    progress_bar.progress(int(10 + (90 * (idx + 1) / total_tasks)))
                except Exception as e:
                    status_box.update(label=f"❌ Failed [{idx+1}/{total_tasks}]: {task_desc}", state="error")
                    st.error(f"Error during subprocess pipeline step: {e}")
                    break
        else:
            st.toast("All tasks completed successfully!", icon="✅")
            st.rerun() # Refresh data exploration state


# ---------------------------------------------------------------------------
# TAB 2: VISUALIZE METRICS
# ---------------------------------------------------------------------------
with tab_viz:
    st.markdown("### 📊 Metrics Comparison Center")
    
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)
            
        # Parse metric JSON into DataFrame
        records = []
        for modality, models in metrics_data.items():
            if models is None:
                continue
            for model_name, metrics in models.items():
                if "error" in metrics:
                    continue
                records.append({
                    "Modality": modality,
                    "Model": model_name,
                    "Accuracy": metrics.get("accuracy", 0.0),
                    "ROC-AUC": metrics.get("roc_auc", 0.0) if isinstance(metrics.get("roc_auc"), float) else 0.5,
                    "Samples": metrics.get("n_samples", 0)
                })
        df_metrics = pd.DataFrame(records)
        
        # 1. Ablation Bar Chart (Accuracy comparison)
        st.markdown("#### Accuracy across Modalities and Models")
        fig_acc = px.bar(
            df_metrics, 
            x="Modality", 
            y="Accuracy", 
            color="Model", 
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text_auto=".3f",
            template="plotly_dark"
        )
        fig_acc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0.4, 1.0]
        )
        st.plotly_chart(fig_acc, use_container_width=True)
        
        # 2. Confusion Matrices & Reports split
        col_sel, col_fig = st.columns([1, 2])
        
        with col_sel:
            st.markdown("#### Detailed reports")
            sel_mod = st.selectbox("Select Modality", options=df_metrics["Modality"].unique())
            sel_model = st.selectbox("Select Classifier Model", options=df_metrics[df_metrics["Modality"] == sel_mod]["Model"].unique())
            
            # Show summary stats
            sub_metric = metrics_data[sel_mod][sel_model]
            st.metric("Accuracy Score", f"{sub_metric['accuracy']:.4f}")
            st.metric("ROC-AUC Score", f"{sub_metric['roc_auc']}")
            st.metric("Folds Tested", f"{sub_metric['cv_folds']}-fold CV")
            
        with col_fig:
            # Heatmap confusion matrix
            cm = np.array(sub_metric["confusion_matrix"])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted Label", y="Actual Label", color="Sample Count"),
                x=["Class 0", "Class 1"],
                y=["Class 0", "Class 1"],
                color_continuous_scale="Viridis",
                title=f"Confusion Matrix ({sel_mod} / {sel_model})",
                template="plotly_dark"
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                width=450,
                height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
    else:
        st.info("No evaluation metrics found. Please execute the pipeline stages to generate metrics.")


# ---------------------------------------------------------------------------
# TAB 3: EXPLORE DATASET
# ---------------------------------------------------------------------------
with tab_explore:
    st.markdown("### 📁 Dataset Inspector")
    
    # Dimensions info box
    if os.path.exists(fusion_meta_file):
        with open(fusion_meta_file, "r") as f:
            f_meta = json.load(f)
            
        st.markdown(
            f"<div class='glass-card'>"
            f"<h4>⚡ Feature Slicing Breakdown</h4>"
            f"Total Fused Feature Dimensions: <b>{f_meta['dim_total']}</b><br>"
            f"Tabular Latent features: <b>{f_meta['dim_tabular']}</b> (Autoencoder representation)<br>"
            f"Image PCA components: <b>{f_meta['dim_image']}</b><br>"
            f"Audio PCA components: <b>{f_meta['dim_audio']}</b><br>"
            f"PCA dimensionality reduction: <b>{'Active' if f_meta['pca_enabled'] else 'Inactive'}</b>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    # Preview csv dataset
    if os.path.exists(final_dataset_csv):
        st.markdown("#### DataFrame Preview (First 50 Rows)")
        df_preview = pd.read_csv(final_dataset_csv, nrows=50)
        
        # Display dataset statistics
        col_rows, col_cols, col_multi = st.columns(3)
        with col_rows:
            st.metric("Total Sample Size", len(pd.read_csv(final_dataset_csv, usecols=["sample_id"])))
        with col_cols:
            st.metric("Total Column Width", len(df_preview.columns))
        with col_multi:
            # Check overlap provenance
            df_prov = pd.read_csv(final_dataset_csv, usecols=["has_tabular", "has_image", "has_audio"])
            multi_count = ((df_prov["has_tabular"].astype(int) + df_prov["has_image"].astype(int) + df_prov["has_audio"].astype(int)) > 1).sum()
            st.metric("Truly Multimodal Samples", multi_count)
            
        st.dataframe(df_preview, height=280)
        
        # Download Deck
        st.markdown("#### 📥 Download Fused Dataset")
        col_csv, col_par = st.columns(2)
        
        with col_csv:
            with open(final_dataset_csv, "rb") as f:
                st.download_button(
                    label="Download Fused CSV Table",
                    data=f,
                    file_name="neurofeature_final_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        with col_par:
            if os.path.exists(final_dataset_parquet):
                with open(final_dataset_parquet, "rb") as f:
                    st.download_button(
                        label="Download Fused Parquet Binary",
                        data=f,
                        file_name="neurofeature_final_dataset.parquet",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
    else:
        st.info("No assembled multimodal dataset found. Run Step 4 (Fuse & Assemble) to construct the combined dataset.")

    # Dynamic Download Deck for All Outputs (Individual & Fused)
    st.markdown("---")
    st.markdown("### 📥 Output Downloads Center")
    
    possible_downloads = {
        "Fused Dataset (CSV)": "neurofeature_final_dataset.csv",
        "Fused Dataset (Parquet)": "neurofeature_final_dataset.parquet",
        "Tabular Features (CSV)": "tabular_final_dataset.csv",
        "Image Features (CSV)": "image_final_dataset.csv",
        "Audio Features (CSV)": "audio_final_dataset.csv",
    }
    
    existing_downloads = {}
    for label, filename in possible_downloads.items():
        f_path = os.path.join(output_dir, filename)
        if os.path.exists(f_path):
            existing_downloads[label] = f_path
            
    if existing_downloads:
        st.markdown("Select and download any generated feature sets:")
        cols = st.columns(len(existing_downloads))
        for i, (label, f_path) in enumerate(existing_downloads.items()):
            with cols[i]:
                with open(f_path, "rb") as f:
                    st.download_button(
                        label=f"⬇️ {label}",
                        data=f,
                        file_name=os.path.basename(f_path),
                        mime="text/csv" if f_path.endswith(".csv") else "application/octet-stream",
                        use_container_width=True
                    )
    else:
        st.warning("⚠️ No processed feature sets or fused tables are available yet. Please upload files and run the pipeline stages (Step 1 to 4) to generate downloadable outputs.")
