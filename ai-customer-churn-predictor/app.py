import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Customer Churn Predictor",
    page_icon="🤖",
    layout="centered"
)

# ---------------- THEME TOGGLE ----------------
theme = st.toggle("🌗 Dark / Light Mode", value=True)

bg_gradient = (
    "linear-gradient(135deg, #0f2027, #203a43, #2c5364)"
    if theme else
    "linear-gradient(135deg, #f5f7fa, #c3cfe2)"
)

text_color = "white" if theme else "#111827"

# ---------------- CUSTOM CSS ----------------
st.markdown(f"""
<style>
body {{
    background: {bg_gradient};
    color: {text_color};
}}
.card {{
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(18px);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 20px 40px rgba(0,0,0,0.4);
}}
.title {{
    font-size: 42px;
    font-weight: 800;
    text-align: center;
}}
.subtitle {{
    text-align: center;
    margin-bottom: 30px;
    opacity: 0.85;
}}
footer {{
    visibility: hidden;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- SAFE FILE CHECK ----------------
REQUIRED_FILES = [
    "model.h5",
    "scaler.pkl",
    "label_encoder_gender.pkl",
    "onehot_encoder_geo.pkl"
]

missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    st.error(f"❌ Missing required files: {', '.join(missing)}")
    st.stop()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model("model.h5")
    with open("label_encoder_gender.pkl", "rb") as f:
        le_gender = pickle.load(f)
    with open("onehot_encoder_geo.pkl", "rb") as f:
        ohe_geo = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_all()

# ---------------- HEADER ----------------
st.markdown("<div class='title'>AI Customer Churn Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>ANN-powered decision system for banking analytics</div>",
    unsafe_allow_html=True
)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
        gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
        age = st.slider("🎂 Age", 18, 92, 35)
        tenure = st.slider("📆 Tenure (Years)", 0, 10, 3)
        credit_score = st.number_input("💳 Credit Score", 300, 900, 600)

    with col2:
        balance = st.number_input("🏦 Account Balance", min_value=0.0, step=1000.0)
        estimated_salary = st.number_input("💰 Estimated Salary", min_value=0.0, step=1000.0)
        num_products = st.slider("📦 Number of Products", 1, 4, 2)
        has_card = st.selectbox("💳 Credit Card", [0, 1])
        is_active = st.selectbox("⚡ Active Member", [0, 1])

    if st.button("🚀 Predict Churn", use_container_width=True):

        gender_encoded = label_encoder_gender.transform([gender])[0]

        base = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [gender_encoded],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_products],
            "HasCrCard": [has_card],
            "IsActiveMember": [is_active],
            "EstimatedSalary": [estimated_salary]
        })

        geo = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_df = pd.DataFrame(
            geo,
            columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
        )

        final_input = pd.concat([base, geo_df], axis=1)
        scaled = scaler.transform(final_input)

        prob = float(model.predict(scaled)[0][0])
        percent = int(prob * 100)

        # ---------- Risk Band ----------
        if percent < 30:
            risk = "🟢 Low Risk"
        elif percent < 60:
            risk = "🟡 Medium Risk"
        else:
            risk = "🔴 High Risk"

        st.markdown("---")
        st.subheader("📊 Prediction Result")
        st.progress(percent)
        st.metric("Churn Probability", f"{percent}%")
        st.write(risk)

        # ---------- Explainability ----------
        st.subheader("🔍 Key Risk Factors")
        reasons = []
        if age > 50: reasons.append("🔺 High Age")
        if tenure < 3: reasons.append("🔺 Low Tenure")
        if balance > 100000: reasons.append("🔺 High Account Balance")
        if num_products == 1: reasons.append("🔺 Single Product Usage")

        if reasons:
            for r in reasons:
                st.write(r)
        else:
            st.write("✅ No major churn risk factors detected")

        # ---------- Business Recommendation ----------
        st.subheader("💡 Recommended Action")
        if percent >= 60:
            st.write("📞 Assign relationship manager & offer loyalty incentives")
        elif percent >= 30:
            st.write("🎯 Send personalized engagement offers")
        else:
            st.write("✅ No action required")

        # ---------- Export ----------
        report = pd.DataFrame({
            "Timestamp": [datetime.now().isoformat()],
            "Churn Probability (%)": [percent],
            "Risk Level": [risk]
        })

        st.download_button(
            "⬇️ Download Report (CSV)",
            report.to_csv(index=False),
            file_name="churn_prediction_report.csv",
            mime="text/csv"
        )

    st.markdown("</div>", unsafe_allow_html=True)
