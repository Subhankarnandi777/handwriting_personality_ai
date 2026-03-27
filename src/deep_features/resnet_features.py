"""
resnet_features.py — Extract deep visual features using a pretrained ResNet-50.

The model is used purely as a feature extractor (classification head removed).
Output: 2048-dimensional feature vector.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from src.utils.config import DEEP, MODELS_PRETRAINED
from src.utils.helper import logger


# ─── Preprocessing transform ─────────────────────────────────────────────────
_TRANSFORM = transforms.Compose([
    transforms.Resize((DEEP["image_size"], DEEP["image_size"])),
    transforms.Grayscale(num_output_channels=3),   # grayscale → 3-channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _build_resnet(device: torch.device) -> nn.Module:
    """Load ResNet-50, strip the FC head, set to eval mode."""
    weights_path = os.path.join(MODELS_PRETRAINED, "resnet50.pth")

    if os.path.exists(weights_path):
        logger.info("Loading ResNet-50 weights from %s", weights_path)
        model = models.resnet50(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        logger.info("Downloading ResNet-50 pretrained weights (ImageNet)…")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        os.makedirs(MODELS_PRETRAINED, exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        logger.info("Saved ResNet-50 weights → %s", weights_path)

    # Remove classification head — keep up to avgpool
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor


# ── Module-level singleton (lazy-loaded) ─────────────────────────────────────
_MODEL = None


def _get_model() -> nn.Module:
    global _MODEL
    if _MODEL is None:
        device = torch.device(DEEP["device"])
        _MODEL = _build_resnet(device)
    return _MODEL


def extract_resnet_features(gray_img: np.ndarray) -> np.ndarray:
    """
    Extract a 2048-D feature vector from a grayscale image array.

    Parameters
    ----------
    gray_img : np.ndarray   shape (H, W) or (H, W, 1)

    Returns
    -------
    np.ndarray  shape (2048,)
    """
    device = torch.device(DEEP["device"])
    model  = _get_model()

    # Convert numpy → PIL
    if gray_img.ndim == 3:
        gray_img = gray_img[:, :, 0]
    pil_img = Image.fromarray(gray_img.astype(np.uint8))

    tensor = _TRANSFORM(pil_img).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    with torch.no_grad():
        features = model(tensor)            # (1, 2048, 1, 1)

    vec = features.squeeze().cpu().numpy()  # (2048,)
    logger.info("ResNet features extracted: shape=%s  norm=%.3f",
                vec.shape, float(np.linalg.norm(vec)))
    return vec
