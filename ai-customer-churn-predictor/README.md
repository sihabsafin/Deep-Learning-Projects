# 🤖 AI Customer Churn Predictor

A production-ready **Artificial Neural Network (ANN)** application that predicts customer churn in the banking sector.  
Built with **TensorFlow** and deployed using **Streamlit**, this project focuses on **real-world inference, explainability, and business decision-making**, not just model training.

🔗 **Live Demo:** https://ai-customer-churn-predictors.streamlit.app/) 

🔗 **LinkedIn:** https://www.linkedin.com/in/sihabsafin/

---

##  Why This Project Matters

Customer churn directly impacts revenue in the banking and fintech industry.  
This application transforms raw customer data into **actionable business insights** by combining:

- ANN-based churn prediction
- Risk interpretation (Low / Medium / High)
- Explainable AI (lightweight, business-friendly)
- Decision-driven recommendations

This is not a demo — it’s a **deployable AI product**.

---

## ✨ Key Features

✅ ANN-based churn probability prediction  
✅ Risk banding (Low 🟢 / Medium 🟡 / High 🔴)  
✅ Animated probability gauge (modern UX)  
✅ Feature contribution insight (mini explainability)  
✅ Business recommendation engine  
✅ Dark / Light mode toggle  
✅ Exportable prediction report (CSV)  
✅ Fully deployed on Streamlit Cloud  

---

## 🧠 Model Overview

- **Algorithm:** Artificial Neural Network (ANN)
- **Framework:** TensorFlow / Keras
- **Task:** Binary Classification (Churn / No Churn)
- **Loss Function:** Binary Crossentropy
- **Output:** Churn Probability (0–100%)

The model is loaded from a pre-trained `.h5` file and used strictly for **inference**, following production best practices.

---

## 🏗️ Tech Stack

- **Python**
- **TensorFlow (ANN)**
- **Scikit-learn**
- **Pandas / NumPy**
- **Streamlit**
- **Streamlit Cloud**

---

## 📂 Project Structure
```bash
ai-customer-churn-predictor/
│
├── app.py # Streamlit application
├── model.h5 # Trained ANN model
├── scaler.pkl # Feature scaler
├── label_encoder_gender.pkl # Gender encoder
├── onehot_encoder_geo.pkl # Geography encoder
│
├── requirements.txt
├── runtime.txt
├── README.md


---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py


