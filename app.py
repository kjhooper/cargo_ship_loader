"""Cargo Ship Loader — Streamlit app entry point.

Run locally:
    conda run -n personal streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Manifest",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("pages/Simulation.py",         title="Cargo Ship Loader",   icon="🚢"),
    st.Page("pages/Benchmarks.py",         title="Benchmarks",          icon="📊"),
    st.Page("pages/Classic_Simulation.py", title="Classic Simulation",  icon="🏛️"),
    st.Page("pages/About.py",              title="About",               icon="ℹ️"),
])
pg.run()
