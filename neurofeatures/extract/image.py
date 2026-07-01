"""
NeuroFeature Pipeline: Image Feature Extractor

This module handles the extraction of general-purpose visual features from diverse 
image datasets. It uses a pretrained MobileNetV2 (ImageNet) running exclusively on 
the CPU to generate fixed-dimensional (1280-dim) global embedding vectors.

These embeddings are designed to be concatenated or fused with text and audio 
modalities downstream in the NeuroFeature pipeline.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, cast
import argparse
import yaml

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image, UnidentifiedImageError

# Configure minimal, clear logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Allowed image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image_paths(root_dir: str) -> List[Path]:
    """
    Recursively collect all valid image files under the given directory.
    
    Args:
        root_dir (str): The root directory to search.
        
    Returns:
        List[Path]: A list of absolute pathlib.Path objects for valid images.
    """
    root_path = Path(root_dir).resolve()
    if not root_path.exists() or not root_path.is_dir():
        logger.error(f"Image directory not found: {root_path}")
        return []

    image_paths = []
    for file_path in root_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            image_paths.append(file_path)
            
    logger.info(f"Found {len(image_paths)} valid image files in {root_path}")
    return image_paths


class MobileNetV2FeatureExtractor:
    """
    General-purpose CPU-based image feature extractor using MobileNetV2.
    Removes the classification head to return raw 1280-dim feature embeddings.
    """
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        
        logger.info("Initializing MobileNetV2 Feature Extractor (CPU)...")
        # Load pretrained model
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.model = mobilenet_v2(weights=weights)
        
        # Strip the final classification layer (Dropout + Linear)
        # MobileNetV2's `classifier` attribute is a torch.nn.Sequential. Replace
        # it with an equivalent Sequential containing a single Identity so that
        # type-checkers and the model forward remain compatible while returning
        # the raw (Batch, 1280) embeddings.
        self.model.classifier = torch.nn.Sequential(torch.nn.Identity())
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        logger.info("Model loaded and set to evaluation mode.")

    def _preprocess_image(self, path: Path) -> Optional[torch.Tensor]:
        """Loads and preprocesses a single image. Handles corruption gracefully."""
        try:
            # .convert('RGB') forces grayscale or alpha channels into standard 3-channel RGB
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                tensor = self.transform(rgb_img)
                # mypy/Type checkers may not infer torchvision.transforms returns a
                # torch.Tensor. Cast explicitly to satisfy static type checks.
                return cast(torch.Tensor, tensor)
        except (UnidentifiedImageError, OSError) as e:
            logger.warning(f"Skipping corrupt or unreadable image: {path} - {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error processing {path}: {e}")
            return None

    @torch.no_grad()
    def extract_features(self, image_paths: List[Path]) -> Dict[str, np.ndarray]:
        """
        Extracts features for a list of image paths using batching.
        
        Args:
            image_paths (List[Path]): List of paths to process.
            
        Returns:
            Dict[str, np.ndarray]: Mapping from absolute file path string to 
                                   1280-dim numpy feature array.
        """
        features_dict: Dict[str, np.ndarray] = {}
        
        batch_tensors = []
        batch_paths = []
        
        total_images = len(image_paths)
        processed_count = 0
        
        logger.info(f"Starting feature extraction in batches of {self.batch_size}...")

        def _process_batch(tensors: List[torch.Tensor], paths: List[str]):
            # Stack into shape (N, C, H, W)
            batch_tensor = torch.stack(tensors).to(self.device)
            # Output shape: (N, 1280)
            embeddings = self.model(batch_tensor).numpy()
            
            for path_str, emb in zip(paths, embeddings):
                features_dict[path_str] = emb

        for idx, path in enumerate(image_paths, 1):
            tensor = self._preprocess_image(path)
            if tensor is not None:
                batch_tensors.append(tensor)
                batch_paths.append(str(path.resolve()))
                
            # Process when batch is full
            if len(batch_tensors) == self.batch_size:
                _process_batch(batch_tensors, batch_paths)
                processed_count += len(batch_paths)
                logger.info(f"Progress: {processed_count}/{total_images} images processed.")
                
                # Clear lists for next batch
                batch_tensors.clear()
                batch_paths.clear()

        # Process any remaining images in the final partial batch
        if batch_tensors:
            _process_batch(batch_tensors, batch_paths)
            processed_count += len(batch_paths)
            logger.info(f"Progress: {processed_count}/{total_images} images processed.")

        logger.info(f"Extraction complete. Successfully extracted features for {processed_count} images.")
        return features_dict


def main():
    """CLI entry point for the NeuroFeature Image Extractor."""
    parser = argparse.ArgumentParser(
        description="Extract features from image datasets"
    )
    parser.add_argument("--config",          default="config.yaml")
    parser.add_argument("--input_dir",       default=None)
    parser.add_argument("--output_feat_dir", default=None)
    args = parser.parse_args()
    
    # Load config
    config = {}
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            
    paths = config.get("paths", {})
    cfg_input = paths.get("image_dir", "neurofeatures/data/images")
    cfg_output = paths.get("image_feat_dir", "neurofeatures/outputs")
    
    image_dir = Path(args.input_dir or cfg_input)
    output_dir = Path(args.output_feat_dir or cfg_output)
    output_file = output_dir / "image_features.npz"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== NeuroFeature: Image Feature Extraction Pipeline ===")
    
    # 1. Discover images
    image_paths = load_image_paths(str(image_dir))
    if not image_paths:
        logger.info("No images to process. Exiting.")
        return

    # 2. Initialize Extractor
    # Using batch_size=16 is generally a safe, memory-friendly sweet spot for CPU
    extractor = MobileNetV2FeatureExtractor(batch_size=16)

    # 3. Extract Features
    features_mapping = extractor.extract_features(image_paths)
    
    if not features_mapping:
        logger.error("No features were extracted. Exiting.")
        return

    # 4. Save to Disk
    # Save as a compressed NumPy archive containing two arrays: 'features' and 'paths'
    logger.info(f"Saving extracted features to {output_file}...")
    
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