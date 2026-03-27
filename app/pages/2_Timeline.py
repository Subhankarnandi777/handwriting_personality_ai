import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px

# Ensure project root is on path so imports work
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.database import get_history

st.set_page_config(page_title="Timeline Tracker", page_icon="📈", layout="wide")

st.markdown("# 📈 Handwriting & Personality Timeline")
st.markdown("Track how your personality traits shift over time based on multiple handwriting samples you've uploaded.")
st.markdown("---")

history = get_history()

if not history:
    st.info("No analyses found in the database. Go to the main page to analyze some handwriting first!")
    st.stop()

df = pd.DataFrame(history)
df['timestamp'] = pd.to_datetime(df['timestamp'])

st.markdown("### Historical Analyses Data")
st.dataframe(df.style.format(precision=2), use_container_width=True)

st.markdown("---")
st.markdown("### 🧬 Trait Evolution")
fig = px.line(
    df, 
    x='timestamp', 
    y=['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'],
    markers=True, 
    title='Big-Five Traits Over Time',
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)
