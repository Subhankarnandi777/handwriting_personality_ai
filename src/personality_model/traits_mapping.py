"""
traits_mapping.py — Graphological rules: features → Big-Five personality scores.

Each rule maps a feature (or combination) to a trait score modifier.
References: Graphology literature + psychology research on handwriting analysis.
"""

from dataclasses import dataclass, field
from typing import Dict

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


# ─── Trait description labels (score-dependent) ───────────────────────────────

TRAIT_DESCRIPTIONS = {
    "Openness": {
        "high":   "Creative, curious, imaginative",
        "medium": "Balanced curiosity and practicality",
        "low":    "Conventional, prefers routine",
    },
    "Conscientiousness": {
        "high":   "Organised, disciplined, reliable",
        "medium": "Generally reliable with some flexibility",
        "low":    "Spontaneous, flexible, sometimes disorganised",
    },
    "Extraversion": {
        "high":   "Sociable, energetic, expressive",
        "medium": "Ambivert — adapts to context",
        "low":    "Reserved, prefers solitude, introspective",
    },
    "Agreeableness": {
        "high":   "Compassionate, cooperative, empathetic",
        "medium": "Generally considerate with healthy assertiveness",
        "low":    "Competitive, direct, sometimes blunt",
    },
    "Neuroticism": {
        "high":   "Emotionally sensitive, prone to anxiety",
        "medium": "Occasional stress with good recovery",
        "low":    "Emotionally stable, calm under pressure",
    },
}


def score_to_label(trait: str, score: float) -> str:
    if score >= 0.60:
        level = "high"
    elif score <= 0.40:
        level = "low"
    else:
        level = "medium"
    return TRAIT_DESCRIPTIONS[trait][level]


# ─── Feature → trait mapping rules ───────────────────────────────────────────

@dataclass
class TraitRule:
    feature_key:   str
    trait:         str
    weight:        float          # signed: positive → increases trait
    description:   str = ""


GRAPHOLOGY_RULES: list[TraitRule] = [

    # ── SLANT ──────────────────────────────────────────────────────────────
    TraitRule("slant_score", "Extraversion",      +0.70,
              "Right slant = sociable/expressive"),
    TraitRule("slant_score", "Agreeableness",     +0.15,
              "Right slant = warm"),
    TraitRule("slant_score", "Neuroticism",       +0.10,
              "Strong right slant = emotional expressiveness"),

    # ── PRESSURE ───────────────────────────────────────────────────────────
    TraitRule("pressure_score", "Conscientiousness", +0.60,
              "Heavy pressure = commitment & vitality"),
    TraitRule("pressure_score", "Neuroticism",       +0.15,
              "Very heavy pressure = emotional intensity"),
    TraitRule("pressure_score", "Openness",          -0.10,
              "Light pressure = sensitivity & openness"),

    # ── LETTER SIZE ────────────────────────────────────────────────────────
    TraitRule("letter_size_score", "Extraversion",      +0.65,
              "Large letters = outgoing, social"),
    TraitRule("letter_size_score", "Openness",          +0.15,
              "Large writing = desire to be noticed"),
    TraitRule("letter_consistency", "Conscientiousness", +0.65,
              "Consistent size = disciplined, reliable"),
    TraitRule("letter_consistency", "Neuroticism",      -0.60,
              "Inconsistency = emotional variability"),

    # ── WORD SPACING ───────────────────────────────────────────────────────
    TraitRule("word_spacing_score", "Openness",       +0.15,
              "Wide spacing = independent thinker"),
    TraitRule("word_spacing_score", "Extraversion",   -0.15,
              "Narrow spacing = sociable, needs closeness"),
    TraitRule("word_spacing_score", "Conscientiousness", +0.10,
              "Wide spacing = organised thinking"),

    # ── LINE SPACING ───────────────────────────────────────────────────────
    TraitRule("line_spacing_score", "Conscientiousness", +0.15,
              "Wide line spacing = clear, organised thought"),
    TraitRule("line_spacing_score", "Openness",          +0.10,
              "Wide spacing = open, uncluttered mind"),

    # ── BASELINE ───────────────────────────────────────────────────────────
    TraitRule("baseline_regularity", "Conscientiousness", +0.20,
              "Straight baseline = stable, reliable"),
    TraitRule("baseline_regularity", "Neuroticism",       -0.70,
              "Wavy baseline = emotional instability"),

    # ── MARGINS ────────────────────────────────────────────────────────────
    TraitRule("left_margin_score",  "Extraversion",      +0.10,
              "Narrow left margin = sociable, forward-looking"),
    TraitRule("right_margin_score", "Neuroticism",       +0.10,
              "Wide right margin = fear of the future"),
    TraitRule("left_margin_score",  "Conscientiousness", +0.10,
              "Consistent left margin = organised"),

    # ── STROKE WIDTH VARIATION ─────────────────────────────────────────────
    TraitRule("stroke_width_std",   "Openness",     +0.15,
              "Variable strokes = artistic, creative"),

    # ── SIGNATURE METRICS ──────────────────────────────────────────────────
    TraitRule("sig_underline",      "Extraversion",      +0.20,
              "Presence of underline = desire for recognition, confidence"),
    TraitRule("sig_underline",      "Conscientiousness", +0.10,
              "Presence of underline = self-reliance, firmness"),
    TraitRule("sig_flourish",       "Openness",          +0.25,
              "Elaborate flourishes = creativity, flair"),
    TraitRule("sig_flourish",       "Extraversion",      +0.15,
              "Elaborate flourishes = desire to be noticed"),
    TraitRule("sig_aspect_ratio",   "Extraversion",      +0.10,
              "High width-to-height ratio = expansive, outgoing nature"),
]
