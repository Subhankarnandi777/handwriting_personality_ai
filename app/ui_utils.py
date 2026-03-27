"""
ui_utils.py — Streamlit helper components and styling for the web app.
"""

import base64
import io
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image


# ─── Constants ────────────────────────────────────────────────────────────────

TRAIT_COLORS = {
    "Openness":          "#6C63FF",
    "Conscientiousness": "#43AA8B",
    "Extraversion":      "#F9C74F",
    "Agreeableness":     "#F8961E",
    "Neuroticism":       "#F94144",
}

TRAIT_ICONS = {
    "Openness":          "🎨",
    "Conscientiousness": "📋",
    "Extraversion":      "🌟",
    "Agreeableness":     "🤝",
    "Neuroticism":       "🌊",
}


# ─── CSS injection ────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .main { background: #0C0C14; }

    .trait-card {
        background: linear-gradient(135deg, #14141F 0%, #1A1A2E 100%);
        border: 1px solid #2A2A40;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    .trait-card:hover { border-color: #6C63FF; }

    .trait-name {
        font-size: 1.05rem;
        font-weight: 600;
        color: #E0E0FF;
        margin-bottom: 4px;
    }
    .trait-label {
        font-size: 0.82rem;
        color: #9090B8;
        margin-bottom: 10px;
    }
    .progress-bg {
        background: #1F1F35;
        border-radius: 100px;
        height: 10px;
        overflow: hidden;
    }
    .progress-fill {
        height: 10px;
        border-radius: 100px;
        transition: width 0.6s ease;
    }

    .score-badge {
        float: right;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #E0E0FF;
        font-weight: 700;
    }

    .feature-chip {
        display: inline-block;
        background: #1A1A30;
        border: 1px solid #2A2A45;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        color: #9090C8;
        margin: 3px;
        font-family: 'JetBrains Mono', monospace;
    }

    .dominant-banner {
        background: linear-gradient(135deg, #1A1A35, #252545);
        border: 2px solid #6C63FF;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 24px;
    }
    .dominant-title { font-size: 0.85rem; color: #9090B8; letter-spacing: 2px; text-transform: uppercase; }
    .dominant-trait { font-size: 2rem; font-weight: 700; color: #FFFFFF; margin: 6px 0; }
    .dominant-desc  { font-size: 0.9rem; color: #B0B0D8; }

    .info-box {
        background: #12121C;
        border-left: 3px solid #6C63FF;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #9090B8;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Component renderers ──────────────────────────────────────────────────────

def render_trait_card(trait: str, score: float, label: str) -> None:
    color    = TRAIT_COLORS.get(trait, "#888")
    icon     = TRAIT_ICONS.get(trait, "")
    pct      = int(score * 100)

    st.markdown(f"""
    <div class="trait-card">
      <div class="trait-name">
        {icon}&nbsp; {trait}
        <span class="score-badge">{score:.2f}</span>
      </div>
      <div class="trait-label">{label}</div>
      <div class="progress-bg">
        <div class="progress-fill" style="width:{pct}%; background:{color};"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_dominant_banner(trait: str, label: str) -> None:
    icon = TRAIT_ICONS.get(trait, "")
    st.markdown(f"""
    <div class="dominant-banner">
      <div class="dominant-title">Dominant Personality Trait</div>
      <div class="dominant-trait">{icon} {trait}</div>
      <div class="dominant-desc">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_chips(features: dict) -> None:
    scalar_keys = [
        "slant_angle_deg", "slant_direction", "pressure_level",
        "baseline_direction", "avg_word_spacing", "avg_letter_spacing",
        "letter_avg_height", "left_margin_ratio",
    ]
    chips_html = ""
    for k in scalar_keys:
        v = features.get(k)
        if v is not None:
            chips_html += f'<span class="feature-chip">{k}: {v}</span>'
    st.markdown(chips_html, unsafe_allow_html=True)


def render_rules_expander(rules: list) -> None:
    with st.expander("🔍 Explainability — Fired Rules"):
        for r in rules:
            effect_color = "#43AA8B" if r["weight"] > 0 else "#F94144"
            st.markdown(
                f"<span style='color:{effect_color};font-weight:600'>"
                f"[{r['trait']}]</span> &nbsp;"
                f"<code>{r['feature']}</code> = {r['value']:.3f} &nbsp;·&nbsp; "
                f"weight <b>{r['weight']:+.2f}</b> → {r['effect']}<br>"
                f"<small style='color:#6060A0;'>{r['reasoning']}</small>",
                unsafe_allow_html=True,
            )
            st.divider()


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """Convert matplotlib figure to PNG bytes for st.image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def pil_to_array(pil_img: Image.Image) -> np.ndarray:
    import cv2
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
