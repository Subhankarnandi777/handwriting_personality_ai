"""
thresholding.py — Convert grayscale handwriting images to binary (ink vs paper).
"""

import cv2
import numpy as np

from src.utils.config import PREPROCESS
from src.utils.helper import logger


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Global Otsu thresholding.
    Works well when the background is uniform.
    Returns a binary image: ink = 255, background = 0.
    """
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    logger.debug("Otsu threshold applied.")
    return binary


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive (local) Gaussian thresholding.
    Better than Otsu when lighting varies across the page.
    Returns a binary image: ink = 255, background = 0.
    """
    block = PREPROCESS["adaptive_block_size"]
    C     = PREPROCESS["adaptive_C"]
    if block % 2 == 0:
        block += 1  # must be odd

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, C
    )
    logger.debug("Adaptive threshold applied (block=%d, C=%d).", block, C)
    return binary


def morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """
    Light morphological opening to remove small noise dots,
    followed by closing to fill tiny ink gaps.
    """
    k   = PREPROCESS["morph_kernel_size"]
    krnl = cv2.getStructuringElement(cv2.MORPH_RECT, k)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  krnl)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, krnl)
    logger.debug("Morphological cleanup applied.")
    return closed


def threshold_image(gray: np.ndarray) -> np.ndarray:
    """
    Select threshold method from config and clean up the result.
    Returns cleaned binary image.
    """
    method = PREPROCESS["threshold_method"]
    if method == "otsu":
        binary = otsu_threshold(gray)
    elif method == "adaptive":
        binary = adaptive_threshold(gray)
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    cleaned = morphological_cleanup(binary)
    logger.info("Thresholding complete (%s). Non-zero pixels: %d",
                method, np.count_nonzero(cleaned))
    return cleaned
