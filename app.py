import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression

st.title("🚢 Titanic Survival Predictor")
st.subheader("🧍 Enter Passenger Details")

# --- Load model safely ---
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    st.error("❌ Model file not found. Please make sure 'model.pkl' is in the same folder as app.py.")
    st.stop()

# --- Collect user input ---
Pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
Sex = st.radio("Sex", ["male", "female"])
Age = st.slider("Age", min_value=0, max_value=80, value=30)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
Fare = st.number_input("Ticket Fare", min_value=0.0, max_value=600.0, value=50.0)
Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# --- Prepare input for model ---
input_df = pd.DataFrame([{
    "Pclass": Pclass,
    "Sex": 0 if Sex == "male" else 1,
    "Age": Age,
    "SibSp": SibSp,
    "Parch": Parch,
    "Fare": Fare,
    "Embarked": {"S": 0, "C": 1, "Q": 2}[Embarked]
}])

# --- Predict ---
if st.button("🚀 Predict Survival"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    st.markdown("### 🎯 Prediction Result")
    if prediction == 1:
        st.success(f"🧍 Survived with {probability:.2%} confidence")
    else:
        st.error(f"💀 Did not survive (Confidence: {probability:.2%})")
