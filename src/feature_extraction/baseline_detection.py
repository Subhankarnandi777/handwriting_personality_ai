"""
baseline_detection.py — Detect the writing baseline and measure regularity.

Graphological interpretation:
  Rising baseline    : optimism, enthusiasm
  Descending baseline: fatigue, pessimism, depression
  Straight baseline  : stability, reliability, self-discipline
  Wavy baseline      : versatility or emotional instability
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression

from src.utils.helper import logger


def detect_baseline(binary: np.ndarray, char_bboxes: list) -> dict:
    """
    Fit a baseline through the bottom edges of character bounding boxes
    using RANSAC (robust to outliers from descenders like g, p, y).

    Returns slope, intercept, direction, and regularity score.
    """
    if len(char_bboxes) < 3:
        return _baseline_default()

    # Use bottom-centre of each bounding box
    xs = np.array([b[0] + b[2] / 2 for b in char_bboxes]).reshape(-1, 1)
    ys = np.array([b[1] + b[3]     for b in char_bboxes])   # bottom edge

    try:
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=RESIDUAL_THRESHOLD,
            min_samples=max(3, int(len(xs) * 0.4)),
            random_state=42,
        )
        ransac.fit(xs, ys)
        slope     = float(ransac.estimator_.coef_[0])
        intercept = float(ransac.estimator_.intercept_)
        inlier_mask = ransac.inlier_mask_
    except Exception as exc:
        logger.warning("RANSAC baseline failed: %s — using linear regression.", exc)
        lr = LinearRegression().fit(xs, ys)
        slope     = float(lr.coef_[0])
        intercept = float(lr.intercept_)
        inlier_mask = np.ones(len(xs), dtype=bool)

    # ── Baseline direction ────────────────────────────────────────────────
    # Normalise slope by image width for a resolution-independent angle
    img_w = binary.shape[1]
    angle = float(np.degrees(np.arctan(slope * img_w / binary.shape[0])))

    if angle > 3:
        direction = "descending"    # y increases downward in image coords
    elif angle < -3:
        direction = "rising"
    else:
        direction = "straight"

    # ── Regularity: residuals of inliers ─────────────────────────────────
    inlier_ys   = ys[inlier_mask]
    inlier_xs   = xs[inlier_mask].ravel()
    predicted   = slope * inlier_xs + intercept
    residuals   = np.abs(inlier_ys - predicted)
    regularity  = max(0.0, 1.0 - float(np.mean(residuals)) / 30.0)

    result = {
        "baseline_slope":      round(slope,     5),
        "baseline_intercept":  round(intercept, 2),
        "baseline_direction":  direction,
        "baseline_angle_deg":  round(angle,     2),
        "baseline_regularity": round(regularity, 4),
    }
    logger.info("Baseline: direction=%s  angle=%.2f°  regularity=%.3f",
                direction, angle, regularity)
    return result


def _baseline_default() -> dict:
    return {
        "baseline_slope":      0.0,
        "baseline_intercept":  0.0,
        "baseline_direction":  "straight",
        "baseline_angle_deg":  0.0,
        "baseline_regularity": 0.5,
    }


RESIDUAL_THRESHOLD = 8   # px
