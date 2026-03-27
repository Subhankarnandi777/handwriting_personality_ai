"""
spacing_analysis.py — Word spacing, line spacing, and letter spacing features.

Graphological interpretation:
  Wide word spacing  : needs personal space, independent thinker
  Narrow word spacing: sociable, crowds don't bother them
  Wide line spacing  : clarity of thought
  Narrow line spacing: economical, sometimes confused thinking
"""

import numpy as np
from typing import List

from src.utils.helper import safe_divide, logger


def word_spacing_stats(binary: np.ndarray,
                        lines: List[np.ndarray],
                        gap_threshold: int = 15) -> dict:
    """
    For each text line, find horizontal gaps between words and compute
    their statistics.
    """
    all_gaps = []

    for line in lines:
        v_proj = np.sum(line, axis=0)
        gaps   = _find_gaps(v_proj, gap_threshold)
        all_gaps.extend(gaps)

    if not all_gaps:
        return {
            "avg_word_spacing":    0.0,
            "std_word_spacing":    0.0,
            "word_spacing_ratio":  0.0,
            "word_spacing_score":  0.5,
        }

    avg = float(np.mean(all_gaps))
    std = float(np.std(all_gaps))
    img_w = binary.shape[1]

    return {
        "avg_word_spacing":    round(avg, 2),
        "std_word_spacing":    round(std, 2),
        "word_spacing_ratio":  round(avg / img_w, 4),
        "word_spacing_score":  round(min(1.0, avg / 60.0), 4),
    }


def line_spacing_stats(lines: List[np.ndarray]) -> dict:
    """
    Estimate inter-line spacing from line heights.
    """
    if len(lines) < 2:
        return {
            "avg_line_spacing":   0.0,
            "line_spacing_score": 0.5,
        }

    heights = [ln.shape[0] for ln in lines]
    avg_h   = float(np.mean(heights))

    # Approximate: spacing ≈ mean line height (assumes equal ink/gap ratio)
    score = min(1.0, avg_h / 80.0)

    return {
        "avg_line_spacing":   round(avg_h, 2),
        "line_spacing_score": round(score, 4),
    }


def letter_spacing_stats(char_bboxes: list) -> dict:
    """
    Estimate letter spacing from horizontal gaps between character bboxes.
    """
    if len(char_bboxes) < 2:
        return {
            "avg_letter_spacing":   0.0,
            "letter_spacing_score": 0.5,
        }

    # Sort by x coordinate
    sorted_boxes = sorted(char_bboxes, key=lambda b: b[0])
    gaps = []
    for i in range(1, len(sorted_boxes)):
        x_prev_end = sorted_boxes[i - 1][0] + sorted_boxes[i - 1][2]
        x_cur      = sorted_boxes[i][0]
        gap        = x_cur - x_prev_end
        if 0 < gap < 100:           # ignore large jumps (line breaks)
            gaps.append(gap)

    if not gaps:
        return {"avg_letter_spacing": 0.0, "letter_spacing_score": 0.5}

    avg   = float(np.mean(gaps))
    score = min(1.0, avg / 30.0)

    return {
        "avg_letter_spacing":   round(avg, 2),
        "letter_spacing_score": round(score, 4),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_gaps(v_proj: np.ndarray, threshold: int) -> List[float]:
    """Return widths of zero-density columns (word gaps)."""
    gaps, in_gap, start = [], False, 0
    for c, val in enumerate(v_proj):
        if val == 0 and not in_gap:
            start  = c
            in_gap = True
        elif val > 0 and in_gap:
            gap = c - start
            if gap >= threshold:
                gaps.append(float(gap))
            in_gap = False
    return gaps


# ─── Combined entry point ─────────────────────────────────────────────────────

def analyze_spacing(binary: np.ndarray,
                    lines: List[np.ndarray],
                    char_bboxes: list) -> dict:
    feats = {}
    feats.update(word_spacing_stats(binary, lines))
    feats.update(line_spacing_stats(lines))
    feats.update(letter_spacing_stats(char_bboxes))
    logger.info("Spacing analysis: %s", feats)
    return feats
