#step1: load the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle

sns.set_style("whitegrid")

st.set_page_config(
    page_title="System Monitoring Dashboard",
    layout="wide"
)

#step 2: load model & individual instances
import joblib
import streamlit as st

@st.cache_resource
def load_assets():
    model = joblib.load("final_model.pkl")
    component_encoder = joblib.load("component_encoder.pkl")
    event_encoder = joblib.load("event_encoder.pkl")
    level_encoder = joblib.load("level_encoder.pkl")
    
    return model, component_encoder, event_encoder, level_encoder

model, component_encoder, event_encoder, level_encoder = load_assets()

#step 3: header

st.title("INTELLIGENT SYSTEM MONITORING & FAILURE PREDICTION PLATFORM")
st.markdown("---")

#step 4: file upload

st.subheader("📤 Upload Log Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

#step 5: main pipeline
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    # -----------------------------
    # CLEAN COLUMNS
    # -----------------------------
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    st.write("Detected Columns:", list(df.columns))

    # -----------------------------
    # REQUIRED CHECK
    # -----------------------------
    required_cols = ["pid", "content_length", "template_length", "component"]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    # -----------------------------
    # ENCODING (ONLY COMPONENT USED IN MODEL)
    # -----------------------------
    try:
        df["component_encoded"] = component_encoder.transform(df["component"])
    except Exception as e:
        st.error(f"Component Encoding Error: {e}")
        st.stop()

    # -----------------------------
    # FEATURES (EXACTLY AS TRAINED)
    # -----------------------------
    X = df[[
        "pid",
        "content_length",
        "template_length",
        "component_encoded"
    ]]

    # -----------------------------
    # PREDICTION
    # -----------------------------
    df["prediction"] = model.predict(X)

#step 6: metrics

    total_logs = len(df)
    warnings = df["prediction"].sum()

    health = 100 - ((warnings / total_logs) * 100)

    risk = (
        "LOW" if health > 90
        else "MODERATE" if health > 75
        else "HIGH"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("System Health", f"{health:.2f}%")
    col2.metric("Risk Level", risk)
    col3.metric("Total Logs", total_logs)

#step 7: system diagnosis
    st.markdown("---")
    st.subheader("🚨 System Diagnosis")

    if warnings > 0:
        top_component = df[df["prediction"] == 1]["component"].value_counts().index[0]

        st.error("High Risk Detected in System")
        st.write(f"🔴 Primary Issue Source: **{top_component}**")
        st.write("⚠️ Maintenance Required")
    else:
        st.success("System is Healthy")

#step 8: chart 1

    df["status"] = df["prediction"].map({0: "Normal", 1: "Anomaly"})

    st.markdown("---")
    st.subheader("📊 System Overview")

    col1, col2 = st.columns([1, 1])

    #step9: chart1
    with col1:

        status_counts = df["status"].value_counts()

        fig1, ax1 = plt.subplots(figsize=(5, 4))

        bars = ax1.bar(
            status_counts.index,
            status_counts.values,
            color=["#2ecc71", "#e74c3c"]
        )

        ax1.set_title("Normal vs Anomaly Logs")
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Status")

        for bar in bars:
            ax1.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                int(bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9
            )

        st.pyplot(fig1, use_container_width=True)

    #step10: chart2
    with col2:

        all_components = df["component"].value_counts()
        anomaly_counts = df.loc[df["prediction"] == 1, "component"].value_counts()

        combined = pd.concat([all_components, anomaly_counts], axis=1)
        combined.columns = ["total_logs", "anomalies"]
        combined = combined.fillna(0)

        combined["anomalies"] = combined["anomalies"].astype(int)

        combined = combined.sort_values("anomalies", ascending=False).head(8)

        fig2, ax2 = plt.subplots(figsize=(6, 4))

        bars = ax2.barh(
            combined.index,
            combined["anomalies"],
            color="#c0392b"
        )

        ax2.set_title("Top Failure Sources")
        ax2.set_xlabel("Error Count")

        for bar in bars:
            width = bar.get_width()
            ax2.text(
                width + 0.2,
                bar.get_y() + bar.get_height()/2,
                str(int(width)),
                va="center",
                fontsize=9
            )

        st.pyplot(fig2, use_container_width=True)

#step 10: high risk logs

    st.subheader("📄 High Risk Logs")
    st.dataframe(df[df["prediction"] == 1].head(10))

    st.markdown("---")
    st.subheader("Executive Summary")

    st.write(f"System operating at **{health:.2f}% health**.")
    st.write(f"Risk classified as **{risk}**.")
    st.write("Continuous monitoring recommended.")

else:
    st.info("👆 Please upload a CSV file to start analysis.")