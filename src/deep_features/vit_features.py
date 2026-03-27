"""
vit_features.py — Extract deep features using a pretrained Vision Transformer (ViT).

Uses HuggingFace transformers: google/vit-base-patch16-224
Output: 768-dimensional [CLS] token embedding.
"""

import os
import numpy as np
import torch
from PIL import Image
from transformers import ViTModel, ViTImageProcessor

from src.utils.config import DEEP, MODELS_PRETRAINED
from src.utils.helper import logger


# ── Module-level singletons (lazy-loaded) ────────────────────────────────────
_VIT_MODEL     = None
_VIT_PROCESSOR = None


def _load_vit():
    global _VIT_MODEL, _VIT_PROCESSOR
    if _VIT_MODEL is None:
        model_name  = DEEP["vit_model"]
        cache_dir   = os.path.join(MODELS_PRETRAINED, "vit_cache")
        os.makedirs(cache_dir, exist_ok=True)

        logger.info("Loading ViT model: %s …", model_name)
        _VIT_PROCESSOR = ViTImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        _VIT_MODEL = ViTModel.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        _VIT_MODEL.eval()

        # Optionally save weights for offline re-use
        vit_path = os.path.join(MODELS_PRETRAINED, "vit_model.pth")
        if not os.path.exists(vit_path):
            torch.save(_VIT_MODEL.state_dict(), vit_path)
            logger.info("Saved ViT weights → %s", vit_path)


def extract_vit_features(gray_img: np.ndarray) -> np.ndarray:
    """
    Extract a 768-D [CLS] embedding from a grayscale image.

    Parameters
    ----------
    gray_img : np.ndarray   shape (H, W)

    Returns
    -------
    np.ndarray  shape (768,)
    """
    _load_vit()
    device = torch.device(DEEP["device"])
    _VIT_MODEL.to(device)

    # Grayscale → RGB PIL
    if gray_img.ndim == 3:
        gray_img = gray_img[:, :, 0]
    pil_img = Image.fromarray(gray_img.astype(np.uint8)).convert("RGB")

    inputs = _VIT_PROCESSOR(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _VIT_MODEL(**inputs)

    # [CLS] token: shape (1, seq_len, 768) → take index 0
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    vec = cls_embedding.squeeze().cpu().numpy()  # (768,)

    logger.info("ViT features extracted: shape=%s  norm=%.3f",
                vec.shape, float(np.linalg.norm(vec)))
    return vec
