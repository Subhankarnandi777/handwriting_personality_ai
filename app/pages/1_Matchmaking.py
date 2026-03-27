import sys
import os
import tempfile
import streamlit as st
import numpy as np

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.main_pipeline import run_pipeline

st.set_page_config(page_title="Compatibility Match", page_icon="❤️", layout="wide")

st.markdown("# ❤️ Handwriting Compatibility Match")
st.markdown("Upload two handwriting samples to check their personality compatibility based on graphological traits.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🚹 Person A")
    upload_a = st.file_uploader("Upload sample A", type=["jpg", "png", "jpeg"], key="a")

with col2:
    st.markdown("### 🚺 Person B")
    upload_b = st.file_uploader("Upload sample B", type=["jpg", "png", "jpeg"], key="b")

st.markdown("---")

if upload_a and upload_b:
    if st.button("💘 Calculate Compatibility", type="primary", use_container_width=True):
        
        def _process(f):
            suffix = os.path.splitext(f.name)[-1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.getvalue())
                path = tmp.name
            try:
                # We skip deep features for speed during matchmaking unless requested
                res = run_pipeline(path, use_deep_features=False, save_outputs=False)
                return res["personality"]["scores"]
            finally:
                os.unlink(path)

        with st.spinner("Analyzing Person A's handwriting..."):
            scores_a = _process(upload_a)
        
        with st.spinner("Analyzing Person B's handwriting..."):
            scores_b = _process(upload_b)
            
        st.success("Comparison Complete!")
        
        # Calculate compatibility (simple inverse mean absolute difference)
        vec_a = np.array([scores_a[k] for k in scores_a])
        vec_b = np.array([scores_b[k] for k in scores_b])
        
        diff = np.abs(vec_a - vec_b)
        compatibility = max(0, 1 - np.mean(diff)) * 100
        
        st.markdown(f"<h2 style='text-align: center; color: #ff4b4b;'>Compatibility Score: {compatibility:.1f}%</h2>", unsafe_allow_html=True)
        st.progress(compatibility / 100)
        
        st.markdown("### Trait Breakdown")
        for k in scores_a:
            diff_val = abs(scores_a[k] - scores_b[k])
            match_level = max(0, 100 - (diff_val * 100))
            st.markdown(f"- **{k}**: Person A ({scores_a[k]:.2f}) | Person B ({scores_b[k]:.2f}) ➔ **{match_level:.1f}% synergy**")
