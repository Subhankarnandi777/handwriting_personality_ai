"""
helper.py — Shared utility functions.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ─── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str = "handwriting_ai") -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


# ─── File / Path helpers ──────────────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def timestamped_name(stem: str, ext: str) -> str:
    """Return   stem_YYYYMMDD_HHMMSS.ext"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{ts}.{ext.lstrip('.')}"


# ─── Image helpers ────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load image as BGR numpy array; raise if not found."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    logger.info("Loaded image %s  shape=%s", path, img.shape)
    return img


def save_image(img: np.ndarray, path: str) -> None:
    """Save numpy array as image."""
    ensure_dirs(os.path.dirname(path))
    cv2.imwrite(path, img)
    logger.info("Saved image → %s", path)


def resize_keep_aspect(img: np.ndarray, target_width: int) -> np.ndarray:
    """Resize image to target_width, preserving aspect ratio."""
    h, w = img.shape[:2]
    if w == target_width:
        return img
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_LANCZOS4)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR → grayscale, handling already-gray inputs."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─── JSON helpers ─────────────────────────────────────────────────────────────

def save_json(data: dict, path: str) -> None:
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_serialise)
    logger.info("Saved JSON  → %s", path)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_serialise(obj):
    """Handle numpy types when serialising to JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


# ─── Numerical helpers ────────────────────────────────────────────────────────

def normalise_0_1(value: float, low: float, high: float) -> float:
    """Clip-normalise value to [0, 1] given expected range [low, high]."""
    if high == low:
        return 0.5
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default
