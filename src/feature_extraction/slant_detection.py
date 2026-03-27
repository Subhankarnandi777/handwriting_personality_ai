"""
slant_detection.py — Estimate handwriting slant from stroke angles.

Graphological interpretation:
  Right slant  (> +5°)  : sociable, expressive, emotional
  Vertical     (-5..+5°): independent, controlled
  Left slant   (< -5°)  : reserved, introverted
"""

import cv2
import numpy as np
from typing import Tuple

from src.utils.config import FEATURES
from src.utils.helper import logger


def detect_slant(binary: np.ndarray) -> dict:
    """
    Detect the dominant slant angle using the Hough line transform
    on vertical stroke edges.

    Returns:
        slant_angle_deg (float): mean slant angle in degrees
        slant_direction (str):   "right" | "vertical" | "left"
        slant_score (float):     normalised 0-1 score (0=left, 0.5=upright, 1=right)
    """
    # ── Edge detection to emphasise stroke boundaries ──────────────────────
    edges = cv2.Canny(binary, threshold1=50, threshold2=150)

    # ── Probabilistic Hough lines ──────────────────────────────────────────
    min_len = FEATURES["slant_min_line_len"]
    lines   = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / FEATURES["slant_angle_bins"],
        threshold=30,
        minLineLength=min_len,
        maxLineGap=5,
    )

    if lines is None or len(lines) == 0:
        logger.warning("Slant detection: no lines found; defaulting to 0°.")
        return _slant_result(0.0)

    # ── Collect angles of lines that are roughly vertical (45-135°) ────────
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            angle = 90.0
        else:
            angle = np.degrees(np.arctan2(dy, dx))

        # Keep only strokes that are more vertical than horizontal
        abs_angle = abs(angle)
        if 45 <= abs_angle <= 135:
            angles.append(angle)

    if not angles:
        logger.warning("Slant detection: no vertical strokes found.")
        return _slant_result(0.0)

    # ── Dominant angle from trimmed mean ───────────────────────────────────
    angles  = np.array(angles)
    p10, p90 = np.percentile(angles, 10), np.percentile(angles, 90)
    trimmed  = angles[(angles >= p10) & (angles <= p90)]
    dominant = float(np.mean(trimmed)) if len(trimmed) else float(np.mean(angles))

    # Convert: 90° = upright → 0°, rightward tilt → positive
    slant = dominant - 90.0

    logger.info("Slant detection: angle=%.2f°  from %d strokes.", slant, len(angles))
    return _slant_result(slant)


def _slant_result(angle: float) -> dict:
    if angle > 5:
        direction = "right"
        score     = min(1.0, 0.5 + angle / 60.0)
    elif angle < -5:
        direction = "left"
        score     = max(0.0, 0.5 + angle / 60.0)
    else:
        direction = "vertical"
        score     = 0.5

    return {
        "slant_angle_deg": round(angle, 2),
        "slant_direction": direction,
        "slant_score":     round(score, 4),
    }
