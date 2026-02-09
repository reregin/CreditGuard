import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="CreditGuard Dashboard", layout="wide")

st.title("🛡️ CreditGuard: AI-Powered Risk Assessment")
st.markdown("### Intelligent Credit Scoring & Fraud Detection System")

# Placeholder for future metrics
col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", "94.2%", "+1.2%")
col2.metric("Fraud Detected", "12", "-2")
col3.metric("Pending Reviews", "5")

st.info("System Ready. Waiting for model integration...")