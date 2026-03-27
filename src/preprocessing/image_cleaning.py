"""
image_cleaning.py — Noise removal and colour normalisation.
"""

import cv2
import numpy as np

from src.utils.config import PREPROCESS
from src.utils.helper import to_gray, resize_keep_aspect, logger


def remove_noise(gray: np.ndarray) -> np.ndarray:
    """
    Apply fast non-local means denoising to a grayscale image.
    Removes scan artefacts, paper texture, and salt-&-pepper noise.
    """
    h = PREPROCESS["denoise_h"]
    denoised = cv2.fastNlMeansDenoising(gray, h=h, templateWindowSize=7, searchWindowSize=21)
    logger.debug("Noise removal applied (h=%s)", h)
    return denoised


def normalise_illumination(gray: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation).
    Corrects uneven lighting across the page.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Straighten a slightly rotated page using the dominant horizontal
    projection profile angle.
    """
    # Binarise for moment computation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return gray  # nothing to deskew

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    # Only correct small tilts (< 10°) to avoid destroying the image
    if abs(angle) > 10:
        logger.debug("Skew angle %.1f° exceeds threshold; skipping deskew", angle)
        return gray

    h, w = gray.shape
    centre   = (w // 2, h // 2)
    rot_mat  = cv2.getRotationMatrix2D(centre, angle, 1.0)
    deskewed = cv2.warpAffine(gray, rot_mat, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    logger.debug("Deskewed image by %.2f°", angle)
    return deskewed


def clean_image(bgr: np.ndarray) -> np.ndarray:
    """
    Full cleaning pipeline:
      1. Resize to standard width
      2. Convert to grayscale
      3. Normalise illumination
      4. Denoise
      5. Deskew

    Returns a clean grayscale image.
    """
    resized  = resize_keep_aspect(bgr, PREPROCESS["resize_width"])
    gray     = to_gray(resized)
    norm     = normalise_illumination(gray)
    denoised = remove_noise(norm)
    result   = deskew(denoised)
    logger.info("Image cleaning complete. Output shape: %s", result.shape)
    return result
