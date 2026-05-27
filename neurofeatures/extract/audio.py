"""
NeuroFeature Pipeline: Audio Feature Extractor (Upgraded)

This module handles the extraction of robust, general-purpose audio embeddings 
from diverse audio files (.wav, .mp3, .flac, .ogg). It acts as the auditory 
counterpart to the `image.py` MobileNetV2 extractor.

It uses a pretrained Wav2Vec 2.0 model running exclusively on the CPU. 
Audio is read directly via Soundfile (bypassing torchaudio codec bugs), 
resampled to 16kHz, converted to mono, and processed to generate a fixed 
768-dim global embedding vector per file.
"""

import os
import gc
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_BASE
import soundfile as sf

# Configure minimal, clear logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Allowed audio extensions
VALID_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

# Audio processing constants
TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 10  # Cap length to prevent CPU OOM on huge files
TARGET_LENGTH_SAMPLES = TARGET_SAMPLE_RATE * MAX_DURATION_SECONDS


def load_audio_paths(root_dir: str) -> List[Path]:
    """Recursively collect all valid audio files under the given directory."""
    root_path = Path(root_dir).resolve()
    if not root_path.exists() or not root_path.is_dir():
        logger.error(f"Audio directory not found: {root_path}")
        return []

    audio_paths = []
    for file_path in root_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            audio_paths.append(file_path)
            
    logger.info(f"Found {len(audio_paths)} valid audio files in {root_path}")
    return audio_paths


class AudioFeatureExtractor:
    """
    General-purpose CPU-based audio feature extractor using Wav2Vec 2.0.
    Mean-pools the temporal feature sequence to return a raw 768-dim embedding.
    """
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        
        logger.info("Initializing Wav2Vec 2.0 Feature Extractor (CPU)...")
        self.bundle = WAV2VEC2_BASE
        self.model = self.bundle.get_model()
        
        self.model.to(self.device)
        self.model.eval()  
        logger.info("Model loaded and set to evaluation mode.")

    def _preprocess_audio(self, path: Path) -> Optional[torch.Tensor]:
        """Loads and preprocesses audio directly via soundfile for stability."""
        try:
            # 1. Direct soundfile bypass (always_2d ensures shape is always [Time, Channels])
            waveform_np, sample_rate = sf.read(str(path), always_2d=True)
            
            # 2. Convert to PyTorch tensor format [Channels, Time] and ensure Float32
            waveform = torch.from_numpy(waveform_np).t().float()
            
            # 3. Convert to mono (average across channels if stereo)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # 4. Resample to standard 16kHz if necessary
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=TARGET_SAMPLE_RATE
                )
                waveform = resampler(waveform)
                
            # 5. Handle Variable Lengths (Pad or Clip)
            num_samples = waveform.shape[1]
            if num_samples > TARGET_LENGTH_SAMPLES:
                waveform = waveform[:, :TARGET_LENGTH_SAMPLES]
            elif num_samples < TARGET_LENGTH_SAMPLES:
                pad_amount = TARGET_LENGTH_SAMPLES - num_samples
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
                
            return waveform.squeeze(0)
            
        except Exception as e:
            logger.warning(f"Skipping corrupt or unreadable audio: {path} - {e}")
            return None

    @torch.no_grad()
    def extract_features(self, audio_paths: List[Path]) -> Dict[str, np.ndarray]:
        """Extracts features in batches and actively manages memory."""
        features_dict: Dict[str, np.ndarray] = {}
        
        batch_tensors = []
        batch_paths = []
        total_audio = len(audio_paths)
        processed_count = 0
        
        logger.info(f"Starting feature extraction in batches of {self.batch_size}...")

        def _process_batch(tensors: List[torch.Tensor], paths: List[str]):
            batch_tensor = torch.stack(tensors).to(self.device)
            
            features, _ = self.model(batch_tensor)
            global_embeddings = torch.mean(features, dim=1).numpy()
            
            for path_str, emb in zip(paths, global_embeddings):
                features_dict[path_str] = emb
                
            # Aggressive memory cleanup after each heavy transformer pass
            del batch_tensor
            del features
            del global_embeddings
            gc.collect()

        for idx, path in enumerate(audio_paths, 1):
            tensor = self._preprocess_audio(path)
            if tensor is not None:
                batch_tensors.append(tensor)
                batch_paths.append(str(path.resolve()))
                
            if len(batch_tensors) == self.batch_size:
                _process_batch(batch_tensors, batch_paths)
                processed_count += len(batch_paths)
                logger.info(f"Progress: {processed_count}/{total_audio} files processed.")
                
                batch_tensors.clear()
                batch_paths.clear()

        # Final partial batch
        if batch_tensors:
            _process_batch(batch_tensors, batch_paths)
            processed_count += len(batch_paths)
            logger.info(f"Progress: {processed_count}/{total_audio} files processed.")

        logger.info(f"Extraction complete. Successfully extracted features for {processed_count} files.")
        return features_dict


def main():
    """CLI entry point for the NeuroFeature Audio Extractor."""
    current_dir = Path(__file__).resolve().parent
    neurofeatures_dir = current_dir.parent
    data_dir = neurofeatures_dir / "data"
    audio_dir = data_dir / "audio"
    output_file = data_dir / "audio_features.npz"
    
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== NeuroFeature: Audio Feature Extraction Pipeline ===")
    
    audio_paths = load_audio_paths(str(audio_dir))
    if not audio_paths:
        logger.info("No audio files to process. Exiting.")
        return

    # Batch size 4 is mathematically safe for CPU memory limits on 10s audio
    extractor = AudioFeatureExtractor(batch_size=4)
    features_mapping = extractor.extract_features(audio_paths)
    
    if not features_mapping:
        logger.error("No features were extracted. Exiting.")
        return

    logger.info(f"Saving extracted audio features to {output_file}...")
    
    paths_list = list(features_mapping.keys())
    features_matrix = np.array(list(features_mapping.values()), dtype=np.float32)
    
    np.savez_compressed(
        output_file, 
        features=features_matrix, 
        paths=paths_list
    )
    
    logger.info("=== Pipeline Complete ===")
    logger.info(f"Final output shape: {features_matrix.shape} saved to disk.")


if __name__ == "__main__":
    main()