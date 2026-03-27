"""
personality_predictor.py — Main predictor that chooses between:
  1. Trained ML model  (personality_model.pkl) — if it exists
  2. Rule engine       — always available fallback

Also provides the unified predict() interface used by main_pipeline.py.
"""

import os
import numpy as np
import joblib
from typing import Dict, Any

from src.utils.config import PERSONALITY
from src.personality_model.rule_engine import (
    apply_rules, get_personality_labels, fired_rules_report
)
from src.personality_model.traits_mapping import TRAITS
from src.utils.helper import logger


# ─── ML model wrapper ─────────────────────────────────────────────────────────

class MLPersonalityModel:
    """Thin wrapper around a scikit-learn pipeline saved with joblib."""

    def __init__(self, path: str):
        self.model  = joblib.load(path)
        self.path   = path
        logger.info("ML personality model loaded from %s", path)

    def predict(self, fused_vec: np.ndarray) -> Dict[str, float]:
        """
        Expect the model to output an array of shape (n_traits,)
        with values in [0, 1].
        """
        vec_2d = fused_vec.reshape(1, -1)
        preds  = self.model.predict(vec_2d)
        scores = preds[0] if preds.ndim > 1 else preds
        return {trait: float(np.clip(scores[i], 0, 1))
                for i, trait in enumerate(TRAITS)}


# ─── Unified Predictor ────────────────────────────────────────────────────────

class PersonalityPredictor:
    def __init__(self):
        model_path = PERSONALITY["model_path"]
        self._ml_model = None

        if os.path.exists(model_path):
            try:
                self._ml_model = MLPersonalityModel(model_path)
                logger.info("Using ML personality model.")
            except Exception as exc:
                logger.warning("Failed to load ML model (%s). Using rule engine.", exc)
        else:
            logger.info("No ML model found at %s. Using rule engine.", model_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        handcrafted_features: Dict[str, Any],
        fused_vec: np.ndarray | None = None,
    ) -> dict:
        """
        Returns:
            scores  : {trait: float 0-1}
            labels  : {trait: str description}
            method  : "ml_model" | "rule_engine"
            rules   : list[dict]  (fired rules for explainability)
        """
        if self._ml_model is not None and fused_vec is not None:
            try:
                scores = self._ml_model.predict(fused_vec)
                method = "ml_model"
            except Exception as exc:
                logger.warning("ML prediction failed (%s). Falling back to rules.", exc)
                scores = apply_rules(handcrafted_features)
                method = "rule_engine_fallback"
        else:
            scores = apply_rules(handcrafted_features)
            method = "rule_engine"

        labels = get_personality_labels(scores)
        rules  = fired_rules_report(handcrafted_features)

        return {
            "scores": scores,
            "labels": labels,
            "method": method,
            "rules":  rules,
        }

    def dominant_trait(self, scores: Dict[str, float]) -> str:
        return max(scores, key=scores.get)

    def personality_summary(self, scores: Dict[str, float],
                             labels: Dict[str, str]) -> str:
        dominant = self.dominant_trait(scores)
        lines = [
            f"Dominant Trait: {dominant}  ({labels[dominant]})",
            "",
            "Big-Five Profile:",
        ]
        for trait in TRAITS:
            bar_len = int(scores[trait] * 20)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {trait:<18} {bar}  {scores[trait]:.2f}  — {labels[trait]}")
        return "\n".join(lines)
