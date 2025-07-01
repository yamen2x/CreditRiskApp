# app.py
# -------------------------------------------------
# Streamlit Credit‑Risk Estimator with XGBoost
# -------------------------------------------------
import os
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image               # <‑‑ for plots
import joblib
from xgboost import XGBClassifier

# -------------------------------------------------
# 1. Load trained XGBoost model
# -------------------------------------------------
MODEL_PATH = "xgb_model.pkl"
FEATURES = [
    "credit_lines_outstanding",
    "debt_to_income",
    "payment_to_income",
    "years_employed",
    "fico_score",
]

@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

model = load_model()

st.set_page_config(page_title="Credit Risk Estimator", layout="centered")
st.title("💳 Credit‑Risk Estimator")

st.write(
    """
This tool estimates **Probability of Default (PD)** and **Expected Loss** for
individual loans or batches of borrowers using an XGBoost classifier trained on the provided loan data.
"""
)

# -------------------------------------------------
# 2. Single‑borrower prediction
# -------------------------------------------------
st.header("🔢 Single Borrower Input")

col1, col2 = st.columns(2)
with col1:
    credit_lines = st.number_input("Credit Lines Outstanding", min_value=0, value=5)
    years_employed = st.number_input("Years Employed", min_value=0, value=3)
    fico = st.number_input("FICO Score", min_value=300, max_value=850, value=700)

with col2:
    debt_to_income = st.slider("Debt‑to‑Income Ratio", 0.0, 1.0, 0.40, step=0.01)
    payment_to_income = st.slider("Payment‑to‑Income Ratio", 0.0, 1.0, 0.20, step=0.01)
    loan_amt = st.number_input("Loan Amount ($)", min_value=1000, step=500, value=15000)

if st.button("Estimate Risk"):
    single = pd.DataFrame(
        [{
            "credit_lines_outstanding": credit_lines,
            "debt_to_income": debt_to_income,
            "payment_to_income": payment_to_income,
            "years_employed": years_employed,
            "fico_score": fico,
        }]
    )
    pd_est = model.predict_proba(single)[0, 1]
    exp_loss = pd_est * 0.9 * loan_amt  # 10 % recovery assumption

    st.success(f"Probability of Default: **{pd_est:.2%}**")
    st.info(f"Expected Loss: **${exp_loss:,.2f}**")

    # optional: save result
    if st.checkbox("Save this prediction"):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = single.copy()
        row["loan_amount"] = loan_amt
        row["PD"] = pd_est
        row["expected_loss"] = exp_loss
        row["timestamp"] = now
        file_exists = os.path.exists("saved_results.csv")
        row.to_csv("saved_results.csv", mode="a", header=not file_exists, index=False)
        st.success("Saved to **saved_results.csv**")

# -------------------------------------------------
# 3. Batch prediction via CSV upload
# -------------------------------------------------
st.header("📁 Batch Prediction")

uploaded = st.file_uploader(
    "Upload a CSV containing borrower data "
    "(must include the columns shown in the example template).",
    type="csv",
)

if uploaded is not None:
    batch = pd.read_csv(uploaded)

    # Ensure required columns / create engineered features
    if "payment_to_income" not in batch.columns:
        batch["payment_to_income"] = (
            batch["loan_amt_outstanding"] / batch["income"]
        )
    if "debt_to_income" not in batch.columns:
        batch["debt_to_income"] = (
            batch["total_debt_outstanding"] / batch["income"]
        )

    # Predict PD & Expected Loss
    batch["PD"] = model.predict_proba(batch[FEATURES])[:, 1]
    batch["Expected_Loss"] = batch["PD"] * 0.9 * batch["loan_amt_outstanding"]

    st.subheader("Preview of Predictions")
    st.dataframe(batch[["customer_id", "PD", "Expected_Loss"]].head())

    # Distribution plot
    st.subheader("PD Distribution")
    fig, ax = plt.subplots()
    sns.histplot(batch["PD"], bins=25, kde=True, ax=ax, color="skyblue")
    ax.set_xlabel("Predicted Probability of Default")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Download button
    csv_bytes = batch.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Full Results",
        data=csv_bytes,
        file_name="batch_predictions.csv",
        mime="text/csv",
    )

# -------------------------------------------------
# 4. FICO Score Distribution Visualizations
# -------------------------------------------------
st.header("📊 FICO Score Bucket Visualizations")

colA, colB = st.columns(2)

with colA:
    st.subheader("K‑Means Buckets")
    st.image(Image.open("figures/ficobuckets_hist.png"),
             caption="K‑Means Buckets", use_container_width=True)

with colB:
    st.subheader("Log‑Likelihood Buckets")
    st.image(Image.open("figures/ficobuckets_ll_hist.png"),
             caption="Log‑Likelihood Buckets", use_container_width=True)

st.subheader("Overall FICO Bucket Distribution")
st.image(Image.open("figures/fico_bucket_distribution.png"),
         caption="All Buckets Combined", use_container_width=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & XGBoost")
