"""
feature_fusion.py — Combine handcrafted features, ResNet, and ViT embeddings.

Strategy
--------
1. Handcrafted features  →  normalised float vector  (N_hc dims)
2. ResNet features       →  L2-normalised             (2048 dims)
3. ViT features          →  L2-normalised             (768 dims)
4. Concatenate all three →  fused vector
5. (Optional) PCA / dimensionality reduction for downstream use
"""

import numpy as np
from typing import Dict, Any

from src.utils.helper import logger


# ─── Keys we keep from the handcrafted feature dict ──────────────────────────
HC_SCALAR_KEYS = [
    "slant_angle_deg",
    "slant_score",
    "avg_word_spacing",
    "word_spacing_score",
    "avg_line_spacing",
    "line_spacing_score",
    "avg_letter_spacing",
    "letter_spacing_score",
    "pressure_mean",
    "pressure_score",
    "stroke_width_mean",
    "stroke_width_std",
    "baseline_angle_deg",
    "baseline_regularity",
    "letter_avg_height",
    "letter_size_score",
    "letter_consistency",
    "left_margin_ratio",
    "right_margin_ratio",
    "top_margin_ratio",
    "bottom_margin_ratio",
]


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _handcrafted_vector(features: Dict[str, Any]) -> np.ndarray:
    """Convert the handcrafted feature dict to a fixed-length float vector."""
    vec = []
    for key in HC_SCALAR_KEYS:
        val = features.get(key, 0.0)
        try:
            vec.append(float(val))
        except (TypeError, ValueError):
            vec.append(0.0)
    return np.array(vec, dtype=np.float32)


def fuse_features(
    handcrafted_features: Dict[str, Any],
    resnet_vec: np.ndarray,
    vit_vec: np.ndarray,
    use_deep: bool = True,
) -> np.ndarray:
    """
    Produce a single fused feature vector.

    Parameters
    ----------
    handcrafted_features : dict from feature extraction modules
    resnet_vec           : (2048,) numpy array
    vit_vec              : (768,)  numpy array
    use_deep             : if False, returns only handcrafted features

    Returns
    -------
    np.ndarray  concatenated fused vector
    """
    hc_vec = _handcrafted_vector(handcrafted_features)

    if not use_deep:
        logger.info("Feature fusion: handcrafted only (%d dims).", len(hc_vec))
        return hc_vec

    rn_norm  = _l2_normalise(resnet_vec.astype(np.float32))
    vit_norm = _l2_normalise(vit_vec.astype(np.float32))

    fused = np.concatenate([hc_vec, rn_norm, vit_norm])
    logger.info(
        "Feature fusion: hc=%d  resnet=%d  vit=%d  total=%d dims",
        len(hc_vec), len(rn_norm), len(vit_norm), len(fused),
    )
    return fused


def reduce_dimensions(fused: np.ndarray, n_components: int = 64) -> np.ndarray:
    """
    Optional PCA dimensionality reduction.
    Useful for downstream ML models that prefer fewer, decorrelated features.
    NOTE: Without a fitted PCA object this performs random-projection as a
    fast approximation.
    """
    if len(fused) <= n_components:
        return fused

    # Deterministic random projection (no training data needed)
    rng   = np.random.default_rng(seed=42)
    proj  = rng.standard_normal((len(fused), n_components)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    reduced = fused @ proj
    logger.info("Dimensionality reduction: %d → %d", len(fused), n_components)
    return reduced
