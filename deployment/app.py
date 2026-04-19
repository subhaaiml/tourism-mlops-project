
import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Prediction", page_icon="✈️", layout="centered")

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "subhaspace/TourismPackage")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "model.pkl")
ENCODER_FILENAME = os.getenv("ENCODER_FILENAME", "label_encoders.pkl")

@st.cache_resource
def load_artifacts():
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type="model")
    enc_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=ENCODER_FILENAME, repo_type="model")
    model = joblib.load(model_path)
    encoders = joblib.load(enc_path)
    return model, encoders

def encode_input(df, encoders):
    data = df.copy()
    for col, encoder in encoders.items():
        if col in data.columns:
            value = str(data.loc[0, col])
            classes = list(encoder.classes_)
            if value not in classes:
                value = classes[0]
            data[col] = encoder.transform([value])[0]
    return data

st.title("Tourism Package Purchase Prediction")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry", "Self Inquiry"])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    preferredpropertystar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
    maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    numberoftrips = st.number_input("Number of Trips", min_value=0, max_value=20, value=2)
    passport = st.selectbox("Passport", [0, 1])
    owncar = st.selectbox("Own Car", [0, 1])
    numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=6, value=0)
    designation = st.selectbox("Designation", ["Manager", "Senior Manager", "AVP", "VP", "Executive"])
    monthlyincome = st.number_input("Monthly Income", min_value=1000, max_value=500000, value=30000)
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    numberoffollowups = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)
    durationofpitch = st.number_input("Duration of Pitch", min_value=0, max_value=100, value=15)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeofcontact,
        "CityTier": citytier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": numberofpersonvisiting,
        "PreferredPropertyStar": preferredpropertystar,
        "MaritalStatus": maritalstatus,
        "NumberOfTrips": numberoftrips,
        "Passport": passport,
        "OwnCar": owncar,
        "NumberOfChildrenVisiting": numberofchildrenvisiting,
        "Designation": designation,
        "MonthlyIncome": monthlyincome,
        "PitchSatisfactionScore": pitchsatisfactionscore,
        "ProductPitched": productpitched,
        "NumberOfFollowups": numberoffollowups,
        "DurationOfPitch": durationofpitch
    }])

    model, encoders = load_artifacts()
    processed_df = encode_input(input_df, encoders)
    prediction = model.predict(processed_df)[0]

    if prediction == 1:
        st.success("This customer is likely to purchase the Wellness Tourism Package.")
    else:
        st.warning("This customer is unlikely to purchase the Wellness Tourism Package.")

    st.dataframe(input_df)
