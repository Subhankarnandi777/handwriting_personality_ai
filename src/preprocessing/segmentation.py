"""
segmentation.py — Line, word, and character region segmentation.
"""

import cv2
import numpy as np
from typing import List, Tuple

from src.utils.helper import logger


# ─── Type aliases ─────────────────────────────────────────────────────────────
BBox = Tuple[int, int, int, int]   # x, y, w, h


# ─── Line segmentation ────────────────────────────────────────────────────────

def segment_lines(binary: np.ndarray) -> List[np.ndarray]:
    """
    Split the binary image into individual text lines using
    horizontal projection profile valleys.

    Returns a list of line sub-images.
    """
    # Horizontal projection: count ink pixels per row
    h_proj = np.sum(binary, axis=1)

    # Find row bands where ink density exceeds threshold
    ink_rows = h_proj > (binary.shape[1] * 0.01)   # >1 % of width is ink

    lines, in_line = [], False
    start = 0
    for r, is_ink in enumerate(ink_rows):
        if is_ink and not in_line:
            start   = r
            in_line = True
        elif not is_ink and in_line:
            if r - start > 5:                       # skip tiny slivers
                lines.append(binary[start:r, :])
            in_line = False
    if in_line:
        lines.append(binary[start:, :])

    logger.info("Segmented %d lines.", len(lines))
    return lines


# ─── Word segmentation ────────────────────────────────────────────────────────

def segment_words(line_img: np.ndarray,
                  gap_threshold: int = 15) -> List[np.ndarray]:
    """
    Split a line image into word blobs using vertical projection valleys.

    gap_threshold: minimum column gap (in px) to consider as word boundary.
    Returns a list of word sub-images.
    """
    v_proj  = np.sum(line_img, axis=0)
    in_word = False
    words, start = [], 0

    for c, density in enumerate(v_proj):
        if density > 0 and not in_word:
            start   = c
            in_word = True
        elif density == 0 and in_word:
            gap = _gap_width(v_proj, c)
            if gap >= gap_threshold:
                words.append(line_img[:, start:c])
                in_word = False
    if in_word:
        words.append(line_img[:, start:])

    return words


def _gap_width(v_proj: np.ndarray, start: int) -> int:
    """Count consecutive zero columns from start."""
    w = 0
    for i in range(start, len(v_proj)):
        if v_proj[i] == 0:
            w += 1
        else:
            break
    return w


# ─── Connected component segmentation ────────────────────────────────────────

def segment_characters(binary: np.ndarray,
                        min_area: int = 50) -> List[BBox]:
    """
    Find character-level bounding boxes via connected components.
    Filters out tiny noise blobs smaller than min_area.

    Returns list of (x, y, w, h) bounding boxes.
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    bboxes = []
    for i in range(1, num_labels):          # skip background label 0
        x, y, w, h, area = stats[i]
        if area >= min_area:
            bboxes.append((int(x), int(y), int(w), int(h)))

    logger.debug("Found %d character-level components.", len(bboxes))
    return bboxes


# ─── Full segmentation ────────────────────────────────────────────────────────

def segment_all(binary: np.ndarray) -> dict:
    """
    Convenience wrapper: returns dict with lines, word counts per line,
    and character bounding boxes.
    """
    lines          = segment_lines(binary)
    word_counts    = [len(segment_words(ln)) for ln in lines]
    char_bboxes    = segment_characters(binary)

    return {
        "lines":          lines,
        "word_counts":    word_counts,
        "num_lines":      len(lines),
        "avg_words_line": float(np.mean(word_counts)) if word_counts else 0.0,
        "char_bboxes":    char_bboxes,
        "num_chars":      len(char_bboxes),
    }
