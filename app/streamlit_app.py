"""
streamlit_app.py — Web UI for the Handwriting Personality AI.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import os
import io
import tempfile
import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Ensure project root is on path ────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.main_pipeline import run_pipeline
from src.utils.visualization import plot_full_analysis, plot_personality_radar
from app.ui_utils import (
    inject_css,
    render_dominant_banner,
    render_trait_card,
    render_feature_chips,
    render_rules_expander,
    fig_to_bytes,
    pil_to_array,
    TRAIT_COLORS,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Handwriting Personality AI",
    page_icon   = "✍️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

inject_css()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✍️ Handwriting AI")
    st.markdown("---")

    st.markdown("### ⚙️ Analysis Options")
    use_deep = st.toggle(
        "Use Deep Learning Features",
        value=False,
        help="ResNet-50 + ViT embeddings. Richer features but requires PyTorch models.",
    )
    is_signature = st.toggle(
        "Signature Analysis Mode",
        value=False,
        help="Enable specialised feature extraction for signatures (flourishes, underlines)."
    )
    save_outputs = st.toggle("Save results to disk", value=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app analyses handwriting images and predicts the **Big-Five** personality traits:

    - 🎨 **Openness**
    - 📋 **Conscientiousness**
    - 🌟 **Extraversion**
    - 🤝 **Agreeableness**
    - 🌊 **Neuroticism**

    Based on graphology research and computer vision feature extraction.
    """)

    st.markdown("---")
    st.caption("Handwriting Personality AI · Built with OpenCV, PyTorch & Streamlit")


# ─── Main content ─────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#E0E0FF; margin-bottom:4px;'>
    ✍️ Handwriting Personality Analysis
</h1>
<p style='text-align:center; color:#6060A0; font-size:1rem; margin-bottom:32px;'>
    Upload a handwriting sample — AI will extract features and predict your Big-Five personality profile.
</p>
""", unsafe_allow_html=True)

# ─── Upload zone ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a handwriting image or PDF",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown("""
    <div style='border:2px dashed #2A2A40; border-radius:16px; padding:48px;
                text-align:center; color:#4A4A70;'>
        <div style='font-size:2.5rem;'>📄</div>
        <div style='font-size:1rem; margin-top:8px;'>
            Drop a handwriting image here or click <b>Browse files</b>
        </div>
        <div style='font-size:0.8rem; margin-top:6px;'>
            JPG · PNG · BMP · TIFF &nbsp;|&nbsp; Recommended: clear scan on white paper
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Run analysis ─────────────────────────────────────────────────────────────
if uploaded_file.name.lower().endswith(".pdf"):
    # Render PDF placeholder
    col_img, col_status = st.columns([2, 1])
    with col_img:
        st.markdown(f"**PDF Document Loaded:** `{uploaded_file.name}`")
        st.info("The application will extract or rasterise the first page of the PDF.")
        # For display, we just show a generic icon or the filename, we can't easily display it natively without rendering
        bgr_array = np.zeros((100, 100, 3), dtype=np.uint8) 
else:
    pil_img   = Image.open(uploaded_file)
    bgr_array = pil_to_array(pil_img)

    col_img, col_status = st.columns([2, 1])
    with col_img:
        st.image(pil_img, caption="Uploaded handwriting")

with col_status:
    st.markdown("### Ready to analyse")
    st.markdown(f"**File:** `{uploaded_file.name}`")
    if not uploaded_file.name.lower().endswith(".pdf"):
        st.markdown(f"**Size:** {pil_img.width} × {pil_img.height} px")
    run_btn = st.button("🔍 Analyse Handwriting", type="primary", use_container_width=True)

if not run_btn:
    st.stop()

# ── Save upload to temp file for pipeline ─────────────────────────────────────
suffix = os.path.splitext(uploaded_file.name)[-1] or ".jpg"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded_file.getvalue())
    tmp_path = tmp.name

try:
    with st.spinner("🔬 Extracting features and predicting personality…"):
        result = run_pipeline(
            image_path        = tmp_path,
            use_deep_features = use_deep,
            save_outputs      = save_outputs,
            is_signature      = is_signature
        )
finally:
    os.unlink(tmp_path)

personality = result["personality"]
scores      = personality["scores"]
labels      = personality["labels"]
features    = result["features"]
elapsed     = result["elapsed_sec"]

# ─── Results header ───────────────────────────────────────────────────────────
st.markdown("---")
dominant_trait = max(scores, key=scores.get)
render_dominant_banner(dominant_trait, labels[dominant_trait])

# ─── Layout: scores + radar ───────────────────────────────────────────────────
col_scores, col_radar = st.columns([3, 2])

with col_scores:
    st.markdown("### Big-Five Personality Scores")
    for trait, score in scores.items():
        render_trait_card(trait, score, labels[trait])

with col_radar:
    st.markdown("### Personality Radar")
    radar_fig  = plot_personality_radar(scores)
    radar_bytes = fig_to_bytes(radar_fig)
    st.image(radar_bytes, use_container_width=True)
    import matplotlib.pyplot as plt
    plt.close(radar_fig)

# ─── Feature chips ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧩 Extracted Handwriting Features")
render_feature_chips(features)

# ─── Full analysis figure ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Full Analysis")
try:
    analysis_fig   = plot_full_analysis(bgr_array, features, scores, labels)
    analysis_bytes = fig_to_bytes(analysis_fig)
    st.image(analysis_bytes, use_container_width=True)
    import matplotlib.pyplot as plt
    plt.close(analysis_fig)
except Exception as exc:
    st.warning(f"Could not render full analysis chart: {exc}")

# ─── Explainability ───────────────────────────────────────────────────────────
render_rules_expander(personality["rules"])

# ─── Raw feature JSON ─────────────────────────────────────────────────────────
with st.expander("📁 Raw Features JSON"):
    serialisable = {
        k: (float(v) if isinstance(v, (float, int)) else str(v))
        for k, v in features.items()
        if not isinstance(v, (list, np.ndarray))
    }
    st.json(serialisable)

# ─── Explainable AI (XAI) View ────────────────────────────────────────────────
if "xai_heatmap" in result["output_paths"]:
    st.markdown("---")
    st.markdown("### 🧠 Deep Learning Explainability (XAI)")
    st.markdown("This pseudo-Grad-CAM heatmap shows which parts of the handwriting excited the ResNet-50 feature extractor the most.")
    xai_path = result["output_paths"]["xai_heatmap"]
    if os.path.exists(xai_path):
        xai_img = Image.open(xai_path)
        # Convert BGR (cv2 write) to RGB
        xai_rgb = cv2.cvtColor(np.array(xai_img), cv2.COLOR_BGR2RGB)
        st.image(xai_rgb, use_container_width=True, caption="ResNet-50 Activation Heatmap")

# ─── Download report ──────────────────────────────────────────────────────────
pdf_path = result["output_paths"].get("report_pdf")
if pdf_path and os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    st.download_button(
        label     = "⬇️ Download Beautiful PDF Report",
        data      = pdf_bytes,
        file_name = os.path.basename(pdf_path),
        mime      = "application/pdf",
    )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; color:#3A3A60; font-size:0.75rem; margin-top:48px;'>
    Analysis completed in {elapsed}s &nbsp;·&nbsp; Method: {personality['method']}
</div>
""", unsafe_allow_html=True)
