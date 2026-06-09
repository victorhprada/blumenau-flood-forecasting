"""Streamlit entry point — redirects to the Overview page."""

import streamlit as st

st.set_page_config(
    page_title="Previsão de Cheias — Blumenau",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.switch_page("pages/1_Overview.py")
