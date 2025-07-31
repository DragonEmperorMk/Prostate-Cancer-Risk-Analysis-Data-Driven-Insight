import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os

st.set_page_config(page_title="Prostate Cancer Risk Analysis", layout="wide")
st.title("🔬 Prostate Cancer Risk Analysis Dashboard")
st.markdown("""
This app performs a **data-driven analysis** of prostate cancer risk using a synthetic dataset.
You can upload your own dataset or use the default sample provided.
""")

# --- Data Upload & Loading ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.convert_dtypes()
    df = df.infer_objects()  # Convert back to standard int, float, etc. if possible

    st.success("✅ File uploaded and loaded successfully!")
else:
    default_path = "synthetic_prostate_cancer_risk.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        df = df.convert_dtypes()
        df = df.infer_objects()  # Convert back to standard int, float, etc. if possible
        st.info("Using default synthetic prostate cancer risk dataset from your local path.")
    else:
        st.error("⚠️ No dataset available. Please upload a file to continue.")

# --- Main App ---
if df is not None:
    # --- Basic Overview ---
    st.header("📊 Dataset Overview")
    st.write(f"**Number of rows:** {df.shape[0]}")
    st.write(f"**Number of columns:** {df.shape[1]}")
    st.dataframe(df.head())

    # --- Data Types & Missing Values ---
    st.subheader("🧼 Data Quality Check")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data Types**")
        st.write(df.dtypes)

    with col2:
        st.markdown("**Missing Values**")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        st.write(missing if not missing.empty else "✅ No missing values!")

    # --- Descriptive Statistics ---
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe())

    # --- Visualizations ---
    st.header("📊 Exploratory Data Analysis")
    if st.checkbox("Show Correlation Heatmap"):
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if st.checkbox("Show Target Variable Distribution"):
        if 'prostate_cancer' in df.columns:
            fig = px.histogram(df, x='prostate_cancer', title='Target Variable Distribution')
            st.plotly_chart(fig)
        else:
            st.warning("Target variable 'prostate_cancer' not found in dataset.")

    # --- Placeholder for Model ---
    st.header("🧠 Risk Prediction (Coming Soon)")
    st.info("A prediction module using machine learning models will be added in the next version.")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")