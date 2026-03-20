import streamlit as st
import pickle
import pandas as pd

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="AI Disease Prediction", layout="centered")

st.title("🧠 AI Heart Disease Prediction")
st.markdown("Enter patient details:")

# INPUTS
col1, col2 = st.columns(2)

with col1:
    age = st.text_input("Age (years)")
    trestbps = st.text_input("Blood Pressure (mm Hg)")
    chol = st.text_input("Cholesterol (mg/dl)")
    thalach = st.text_input("Max Heart Rate")
    oldpeak = st.text_input("Oldpeak")
    ca = st.text_input("Number of vessels (0-3)")

with col2:
    sex = st.selectbox("Gender", ["Male", "Female"])
    dataset = st.selectbox("Dataset", ["Cleveland", "Hungary", "Switzerland", "VA"])
    cp = st.selectbox("Chest Pain", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    fbs = st.selectbox("High Blood Sugar", ["No", "Yes"])
    restecg = st.selectbox("ECG", ["Normal", "ST-T abnormality"])
    exang = st.selectbox("Exercise Angina", ["No", "Yes"])

col3, col4 = st.columns(2)

with col3:
    slope = st.selectbox("Slope", ["Flat", "Upsloping", "Downsloping"])
with col4:
    thal = st.selectbox("Thal", ["Normal", "Reversible Defect", "Fixed Defect"])

# PREDICT
if st.button("Predict"):

    try:
        # Convert inputs
        age = float(age)
        trestbps = float(trestbps)
        chol = float(chol)
        thalach = float(thalach)
        oldpeak = float(oldpeak)
        ca = float(ca)

        # Basic validation (IMPORTANT)
        if thalach > 220 or chol > 400:
            st.error("⚠ Enter realistic medical values")
            st.stop()

        # Encoding
        input_dict = {
            'age': age,
            'trestbps': trestbps,
            'chol': chol,
            'thalch': thalach,
            'oldpeak': oldpeak,
            'ca': ca,
            'sex_Male': 1 if sex == "Male" else 0,
            'dataset_Hungary': 1 if dataset == "Hungary" else 0,
            'dataset_Switzerland': 1 if dataset == "Switzerland" else 0,
            'dataset_VA Long Beach': 1 if dataset == "VA" else 0,
            'cp_atypical angina': 1 if cp == "atypical angina" else 0,
            'cp_non-anginal': 1 if cp == "non-anginal" else 0,
            'cp_typical angina': 1 if cp == "typical angina" else 0,
            'fbs_True': 1 if fbs == "Yes" else 0,
            'restecg_normal': 1 if restecg == "Normal" else 0,
            'restecg_st-t abnormality': 1 if restecg == "ST-T abnormality" else 0,
            'exang_True': 1 if exang == "Yes" else 0,
            'slope_flat': 1 if slope == "Flat" else 0,
            'slope_upsloping': 1 if slope == "Upsloping" else 0,
            'thal_normal': 1 if thal == "Normal" else 0,
            'thal_reversable defect': 1 if thal == "Reversible Defect" else 0
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=columns, fill_value=0)

        prob = model.predict_proba(input_df)[0][1]

        st.divider()

        if prob < 0.4:
            st.success("✅ NO RISK")
        elif prob < 0.7:
            st.warning("⚠ LOW RISK")
        else:
            st.error("🚨 HIGH RISK")

        st.write(f"Risk Probability: {round(prob*100,2)}%")

    except:
        st.warning("⚠ Enter valid numeric values")

st.caption("⚠ This is an academic ML model, not medical advice.")