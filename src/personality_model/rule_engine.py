"""
rule_engine.py — Score Big-Five traits from handcrafted features using
                  the graphological rules in traits_mapping.py.
"""

import numpy as np
from typing import Dict, Any

from src.personality_model.traits_mapping import (
    GRAPHOLOGY_RULES, TRAITS, score_to_label, TraitRule
)
from src.utils.helper import logger


def apply_rules(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Apply all graphology rules to the feature dict and return
    a dict of {trait: score (0-1)}.

    Scores start at 0.50 (neutral) and are adjusted by each rule.
    Sigmoid activation keeps scores in [0, 1].
    """
    # Accumulators
    raw_scores: Dict[str, float] = {t: 0.0 for t in TRAITS}
    weights:    Dict[str, float] = {t: 0.0 for t in TRAITS}

    for rule in GRAPHOLOGY_RULES:
        feat_val = features.get(rule.feature_key)
        if feat_val is None:
            continue
        try:
            val = float(feat_val)
        except (TypeError, ValueError):
            continue

        # Normalise feature value to [0, 1] if it looks like a score already,
        # otherwise clip to a sensible range.
        val_norm = float(np.clip(val, 0.0, 1.0))

        contribution = rule.weight * val_norm
        raw_scores[rule.trait] += contribution
        weights[rule.trait]    += abs(rule.weight)

    # Normalise each trait
    final: Dict[str, float] = {}
    for trait in TRAITS:
        total_w = weights[trait]
        if total_w == 0:
            final[trait] = 0.5
        else:
            # Centre at 0.5, scale contribution
            normalised  = 0.5 + raw_scores[trait] / total_w
            final[trait] = float(np.clip(normalised, 0.0, 1.0))

    logger.info("Rule-engine scores: %s",
                {k: round(v, 3) for k, v in final.items()})
    return final


def get_personality_labels(scores: Dict[str, float]) -> Dict[str, str]:
    """Convert numeric scores to human-readable labels."""
    return {trait: score_to_label(trait, score)
            for trait, score in scores.items()}


def fired_rules_report(features: Dict[str, Any]) -> list[dict]:
    """
    Return a list of rules that fired (feature key was present),
    useful for explainability.
    """
    report = []
    for rule in GRAPHOLOGY_RULES:
        val = features.get(rule.feature_key)
        if val is not None:
            report.append({
                "feature":   rule.feature_key,
                "value":     round(float(val), 4),
                "trait":     rule.trait,
                "weight":    rule.weight,
                "effect":    "increases" if rule.weight > 0 else "decreases",
                "reasoning": rule.description,
            })
    return report
