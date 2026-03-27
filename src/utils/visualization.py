"""
visualization.py — Charts and annotated images for analysis results.
"""

import os
import textwrap

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from src.utils.helper import ensure_dirs, logger


# ─── Colour palette ───────────────────────────────────────────────────────────
TRAIT_COLORS = {
    "Openness":          "#6C63FF",
    "Conscientiousness": "#43AA8B",
    "Extraversion":      "#F9C74F",
    "Agreeableness":     "#F8961E",
    "Neuroticism":       "#F94144",
}
DEFAULT_COLOR = "#888888"


# ─── Feature overlay on image ─────────────────────────────────────────────────

def draw_feature_overlay(image: np.ndarray, features: dict) -> np.ndarray:
    """
    Draw lightweight feature annotations directly on a copy of the image.
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    # ── Baseline line (if detected) ──
    if "baseline_slope" in features and "baseline_intercept" in features:
        slope     = features["baseline_slope"]
        intercept = features["baseline_intercept"]
        x1, x2    = 0, w
        y1 = int(slope * x1 + intercept)
        y2 = int(slope * x2 + intercept)
        cv2.line(vis, (x1, y1), (x2, y2), (0, 200, 0), 2, cv2.LINE_AA)

    # ── Margin guides ──
    if "left_margin_px" in features:
        cv2.line(vis, (features["left_margin_px"], 0),
                 (features["left_margin_px"], h), (255, 100, 0), 2)
    if "right_margin_px" in features:
        cv2.line(vis, (features["right_margin_px"], 0),
                 (features["right_margin_px"], h), (0, 100, 255), 2)

    # ── Slant label ──
    slant = features.get("slant_angle_deg", None)
    if slant is not None:
        label = f"Slant: {slant:.1f} deg"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 0, 200), 2, cv2.LINE_AA)

    return vis


# ─── Personality radar chart ──────────────────────────────────────────────────

def plot_personality_radar(scores: dict, save_path: str | None = None) -> plt.Figure:
    """
    Radar / spider chart for Big-Five personality scores.
    scores: {trait: float 0-1}
    """
    traits = list(scores.keys())
    values = [scores[t] for t in traits]

    N      = len(traits)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles      = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    ax.plot(angles, values_plot, "o-", linewidth=2, color="#6C63FF")
    ax.fill(angles, values_plot, alpha=0.25, color="#6C63FF")

    ax.set_thetagrids(np.degrees(angles[:-1]), traits,
                      color="white", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       color="#888", fontsize=7)
    ax.grid(color="#333", linestyle="--", linewidth=0.5)
    ax.spines["polar"].set_color("#444")

    ax.set_title("Personality Profile", color="white",
                 fontsize=13, pad=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        ensure_dirs(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Saved radar chart → %s", save_path)
    return fig


# ─── Comprehensive analysis figure ───────────────────────────────────────────

def plot_full_analysis(
    image: np.ndarray,
    features: dict,
    personality_scores: dict,
    personality_labels: dict,
    save_path: str | None = None,
) -> plt.Figure:
    """
    4-panel figure:
      [0] Annotated image
      [1] Feature bar chart
      [2] Personality radar
      [3] Personality bar chart with labels
    """
    fig = plt.figure(figsize=(16, 10), facecolor="#0F0F1A")
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 0: annotated image ──────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor("#0F0F1A")
    annotated = draw_feature_overlay(image, features)
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    ax0.imshow(rgb)
    ax0.set_title("Input Image (annotated)", color="white", fontsize=11)
    ax0.axis("off")

    # ── Panel 1: feature bar chart ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor("#0F0F1A")
    numeric_feats = {
        k: float(v)
        for k, v in features.items()
        if isinstance(v, (int, float, np.floating, np.integer))
           and not np.isnan(float(v))
    }
    feat_names  = list(numeric_feats.keys())[:12]
    feat_values = [numeric_feats[k] for k in feat_names]
    bars = ax1.barh(feat_names, feat_values, color="#43AA8B", edgecolor="none")
    ax1.set_title("Extracted Features", color="white", fontsize=11)
    ax1.tick_params(colors="white", labelsize=8)
    ax1.spines[:].set_color("#333")
    ax1.set_facecolor("#0F0F1A")
    for bar, val in zip(bars, feat_values):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", color="white", fontsize=7)

    # ── Panel 2: personality radar ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0], polar=True)
    ax2.set_facecolor("#0F0F1A")
    traits  = list(personality_scores.keys())
    values  = [personality_scores[t] for t in traits]
    N       = len(traits)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    v_plot  = values + [values[0]]
    a_plot  = angles + [angles[0]]
    ax2.plot(a_plot, v_plot, "o-", linewidth=2, color="#6C63FF")
    ax2.fill(a_plot, v_plot, alpha=0.25, color="#6C63FF")
    ax2.set_thetagrids(np.degrees(angles), traits, color="white", fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels([], color="#888")
    ax2.grid(color="#333", linestyle="--", linewidth=0.5)
    ax2.spines["polar"].set_color("#444")
    ax2.set_title("Big-Five Radar", color="white", fontsize=11, pad=14)

    # ── Panel 3: personality bars ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0F0F1A")
    colors = [TRAIT_COLORS.get(t, DEFAULT_COLOR) for t in traits]
    bars3  = ax3.bar(traits, values, color=colors, edgecolor="none", width=0.6)
    ax3.set_ylim(0, 1.15)
    ax3.set_title("Personality Scores", color="white", fontsize=11)
    ax3.tick_params(colors="white", labelsize=9)
    ax3.spines[:].set_color("#333")
    for bar, trait in zip(bars3, traits):
        label = personality_labels.get(trait, "")
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}\n{label}",
            ha="center", va="bottom", color="white", fontsize=7, linespacing=1.3,
        )

    # ── Super-title ───────────────────────────────────────────────────────────
    fig.suptitle("Handwriting Personality Analysis", color="white",
                 fontsize=15, fontweight="bold", y=1.01)

    plt.tight_layout()
    if save_path:
        ensure_dirs(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Saved full analysis → %s", save_path)
    return fig
