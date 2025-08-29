import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Shanmuganathan75/Tourism-Package-Prediction", filename="best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

# --- Streamlit Inputs ---
st.title("Tourism Package Prediction")
st.write("""
This application predicts the likelihood of a Package selection prediction.
Please enter the   data below to get a prediction.
""")
col1, col2, col3 = st.columns(3)
with col1:
# Categorical inputs
  TypeofContact = st.selectbox("Type of Contact ", ["Company Invited", "Self Enquiry"])
  Occupation = st.selectbox("Occuptaion ", ["Salaried","Free Lancer","Small Business","Large Business"]) 
  Gender = st.selectbox("Gender", ["Male", "Female"])
  ProductPitched = st.selectbox("Product Pitched ", ["Basic","Deluxe","King","Standard","Super Deluxe"])
  MaritalStatus = st.selectbox("Marital Status ", ["Single","Married","Unmarried","Divorced"]) 
  Designation = st.selectbox("Designation ", ["VP","AVP","Manager","Senior Manager","Executive"])
with col2:
  # Numerical inputs
  Age = st.number_input("Age", min_value=0, max_value=100, value=40, step=1)  
  CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=2, step=1)  
  DurationOfPitch = st.number_input("Duration of Pitch", min_value=1, max_value=1000, value=20, step=1)
  NumberOfPersonVisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=2, step=1) 
  NumberOfFollowups = st.number_input("Number of Followups", min_value=1, max_value=10, value=2, step=1) 
  PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=2, step=1) 
with col3:
  NumberOfTrips = st.number_input("Number of Trip", min_value=1, max_value=10, value=2, step=1) 
  Passport = st.number_input("Passport Availability", min_value=0, max_value=1, value=1, step=1) 
  PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1) 
  OwnCar = st.number_input("Own Car", min_value=0, max_value=1, value=1, step=1) 
  NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=1, max_value=5, value=3, step=1)
  MonthlyIncome = st.number_input("Monthly Income", min_value=100, max_value=300000, value=15000, step=100)  


# --- Build DataFrame in exact order expected by model ---
input_dict = {
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome
}

input_data = pd.DataFrame([input_dict])

# Ensure column order matches training pipeline
input_data = input_data[model.feature_names_in_]


if st.button("Predict Package"):
    prediction = model.predict(input_data)[0]
    result = "Product Selected" if prediction == 1 else "Product Not selected"
    st.subheader("Tourism Package Result:")
    st.success(f"The model predicts: **{result}**")
