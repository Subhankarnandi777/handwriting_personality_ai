"""
pressure_analysis.py — Estimate pen pressure from pixel intensity.

Graphological interpretation:
  Heavy pressure  : vitality, strong emotions, takes commitments seriously
  Medium pressure : healthy balance
  Light pressure  : sensitivity, empathy, lack of confidence
"""

import numpy as np

from src.utils.config import FEATURES
from src.utils.helper import normalise_0_1, logger


def analyze_pressure(gray: np.ndarray, binary: np.ndarray) -> dict:
    """
    Estimate pen pressure by examining the intensity of ink pixels.

    Strategy:
      - Extract pixel values ONLY at ink locations (binary > 0)
      - Darker values (lower intensity in grayscale) = heavier pressure
      - Compute mean, std, and a normalised pressure score
    """
    # Ink mask: pixels the threshold decided are ink
    ink_mask = binary > 0

    if np.count_nonzero(ink_mask) == 0:
        return {
            "pressure_mean":  0.0,
            "pressure_std":   0.0,
            "pressure_level": "medium",
            "pressure_score": 0.5,
        }

    # In grayscale: 0=black (heavy ink) → 255=white (paper)
    ink_pixels = gray[ink_mask].astype(float)

    # Invert so that darker ink → higher "pressure" value
    pressure_vals = 255.0 - ink_pixels

    mean_pressure = float(np.mean(pressure_vals))
    std_pressure  = float(np.std(pressure_vals))

    # Normalise: 0 = feather-light, 1 = very heavy
    score = normalise_0_1(mean_pressure, low=50, high=220)

    if score > 0.65:
        level = "heavy"
    elif score < 0.35:
        level = "light"
    else:
        level = "medium"

    result = {
        "pressure_mean":  round(mean_pressure, 2),
        "pressure_std":   round(std_pressure,  2),
        "pressure_level": level,
        "pressure_score": round(score,          4),
    }
    logger.info("Pressure analysis: %s", result)
    return result


def stroke_width_variation(binary: np.ndarray) -> dict:
    """
    Estimate variation in stroke width via distance transform.
    High variation → expressive / artistic tendencies.
    """
    import cv2
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ink_dists = dist[binary > 0]

    if len(ink_dists) == 0:
        return {"stroke_width_mean": 0.0, "stroke_width_std": 0.0}

    return {
        "stroke_width_mean": round(float(np.mean(ink_dists)), 3),
        "stroke_width_std":  round(float(np.std(ink_dists)),  3),
    }
