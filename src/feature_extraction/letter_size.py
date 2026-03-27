"""
letter_size.py — Measure letter height, width, and size consistency.

Graphological interpretation:
  Large letters : sociable, good at performing, extroverted
  Small letters : concentrated, detail-oriented, introverted
  Inconsistent  : unpredictability, creativity, or emotional swings
"""

import numpy as np
from typing import List, Tuple

from src.utils.helper import normalise_0_1, logger

BBox = Tuple[int, int, int, int]


def analyze_letter_size(char_bboxes: List[BBox],
                         image_height: int) -> dict:
    """
    Compute height, width, aspect ratio, and size consistency
    from character bounding boxes.
    """
    if not char_bboxes:
        return _default_size()

    heights = np.array([b[3] for b in char_bboxes], dtype=float)
    widths  = np.array([b[2] for b in char_bboxes], dtype=float)

    # Filter extreme outliers (> 3 std from mean height)
    h_mean, h_std = heights.mean(), heights.std()
    mask    = (heights > h_mean - 3 * h_std) & (heights < h_mean + 3 * h_std)
    heights = heights[mask]
    widths  = widths[mask]

    if len(heights) == 0:
        return _default_size()

    avg_h   = float(np.mean(heights))
    std_h   = float(np.std(heights))
    avg_w   = float(np.mean(widths))
    std_w   = float(np.std(widths))

    # Aspect ratio: wide letters > 1, narrow < 1
    safe_w   = np.where(widths > 0, widths, 1)
    aspects  = heights / safe_w
    avg_asp  = float(np.mean(aspects))

    # Relative size (normalised against image height)
    size_norm  = normalise_0_1(avg_h, low=5, high=image_height * 0.2)

    # Consistency: coefficient of variation (lower = more consistent)
    cv          = std_h / avg_h if avg_h > 0 else 1.0
    consistency = max(0.0, 1.0 - cv)

    result = {
        "letter_avg_height":    round(avg_h,       2),
        "letter_std_height":    round(std_h,       2),
        "letter_avg_width":     round(avg_w,       2),
        "letter_std_width":     round(std_w,       2),
        "letter_aspect_ratio":  round(avg_asp,     3),
        "letter_size_score":    round(size_norm,   4),
        "letter_consistency":   round(consistency, 4),
    }
    logger.info("Letter size: avg_h=%.1f  size_score=%.3f  consistency=%.3f",
                avg_h, size_norm, consistency)
    return result


def _default_size() -> dict:
    return {
        "letter_avg_height":   0.0,
        "letter_std_height":   0.0,
        "letter_avg_width":    0.0,
        "letter_std_width":    0.0,
        "letter_aspect_ratio": 1.0,
        "letter_size_score":   0.5,
        "letter_consistency":  0.5,
    }
