"""
margins.py — Detect left, right, top, and bottom margins.

Graphological interpretation:
  Left margin  : attitude towards the past / social norms
  Right margin : attitude towards the future / social contact
  Wide top     : reserved, formal
  Wide bottom  : materialistic tendencies
"""

import numpy as np

from src.utils.helper import normalise_0_1, logger


def detect_margins(binary: np.ndarray) -> dict:
    """
    Estimate margins by finding the first/last ink row/column that
    exceeds a density threshold.

    Returns pixel positions and normalised ratios.
    """
    h, w = binary.shape

    # ── Horizontal projection (rows) ──────────────────────────────────────
    h_proj = np.sum(binary, axis=1) / 255
    # ── Vertical projection (cols) ───────────────────────────────────────
    v_proj = np.sum(binary, axis=0) / 255

    threshold_col = w * 0.01   # >1 % of row width has ink
    threshold_row = h * 0.01

    # Left margin: first column with ink
    left_cols = np.where(v_proj > threshold_col)[0]
    left_px   = int(left_cols[0])  if len(left_cols) else 0

    # Right margin: last column with ink
    right_px  = int(left_cols[-1]) if len(left_cols) else w

    # Top margin: first row with ink
    top_rows  = np.where(h_proj > threshold_row)[0]
    top_px    = int(top_rows[0])   if len(top_rows) else 0

    # Bottom margin: last row with ink
    bottom_px = int(top_rows[-1])  if len(top_rows) else h

    # Normalised ratios
    left_ratio   = round(left_px / w,        4)
    right_ratio  = round((w - right_px) / w, 4)
    top_ratio    = round(top_px / h,         4)
    bottom_ratio = round((h - bottom_px) / h, 4)

    # Scores (0=narrow margin, 1=wide margin)
    result = {
        "left_margin_px":    left_px,
        "right_margin_px":   right_px,
        "top_margin_px":     top_px,
        "bottom_margin_px":  bottom_px,
        "left_margin_ratio":   left_ratio,
        "right_margin_ratio":  right_ratio,
        "top_margin_ratio":    top_ratio,
        "bottom_margin_ratio": bottom_ratio,
        "left_margin_score":   round(normalise_0_1(left_ratio,   0, 0.25), 4),
        "right_margin_score":  round(normalise_0_1(right_ratio,  0, 0.25), 4),
    }
    logger.info("Margins: left=%dpx  right=%dpx  top=%dpx  bottom=%dpx",
                left_px, w - right_px, top_px, h - bottom_px)
    return result
