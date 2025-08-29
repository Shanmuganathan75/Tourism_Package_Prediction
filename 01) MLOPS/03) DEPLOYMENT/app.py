import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Shanmuganathan75/Tourism-Package-Prediction", filename="best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism  Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a Package selection prediction.
Please enter the   data below to get a prediction.
""")
# Tourism Predictor Input
CustomerID=st.number_input("Customer ID", min_value=200000, max_value=300000, value=200000, step=1)  
Age=st.number_input("Age", min_value=0, max_value=100, value=40, step=1)  
TypeofContact=st.selectbox("Type of Contact ", ["Company Invited", "Self Enquiry"])
CityTier=st.number_input("City Tier", min_value=1, max_value=3, value=2, step=1)  
DurationOfPitch=st.number_input("Duration of Pitch", min_value=1, max_value=1000, value=20, step=1)
Occupation=st.selectbox("Occuptaion ", ["Salaried","Free Lancer","Small Business","Large Business"]) 
Gender=st.selectbox("Gender ", ["Male","Female"])    
NumberOfPersonVisiting=st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=2, step=1) 
NumberOfFollowups=st.number_input("Number of Followups", min_value=1, max_value=10, value=2, step=1) 
ProductPitched= st.selectbox("Product Pitched ", ["Basic","Deluxe","King","Standard","Super Deluxe"])
PreferredPropertyStar=st.number_input("Preferred Property Star", min_value=1, max_value=5, value=2, step=1) 
MaritalStatus= st.selectbox("Marital Status ", ["Single","Married","Unmarried","Divorced"]) 
NumberOfTrips=st.number_input("Number of Trip", min_value=1, max_value=10, value=2, step=1) 
Passport=st.number_input("Passport Availability", min_value=0, max_value=1, value=1, step=1) 
PitchSatisfactionScore=st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1) 
OwnCar=st.number_input("Own Car", min_value=0, max_value=1, value=1, step=1) 
NumberOfChildrenVisiting=st.number_input("Number of Children Visiting", min_value=1, max_value=5, value=3, step=1) 
Designation=st.selectbox("Designation ", ["VP","AVP","Manager","Senior Manager","Executive"])
MonthlyIncome=st.number_input("Monthly Income", min_value=100, max_value=300000, value=15000, step=100)  

# Assemble input into DataFrame
input_data = pd.DataFrame([{
      'Customer_ID':CustomerID,
      'Age':Age,
      'Type_of_contact':TypeofContact,
      'City_Tier' :CityTier,
      'Duration_of_pitch':DurationOfPitch,
      'Occupation':Occupation,
      'Gender':Gender,
      'Number_of+person_visting':NumberOfPersonVisiting,
      'Number of Followups':NumberOfFollowups,
      'Product Pitched':ProductPitched,
      'Preferred_Property_Star': PreferredPropertyStar,
      'Marital_Status':MaritalStatus,
      'Number_of_Trips':NumberOfTrips,
      'Passpor':Passport,
      'Pitch_Satisfacton_score':PitchSatisfactionScore,
      'Own_car':OwnCar,
      'Number_of_Children_visiting':NumberOfChildrenVisiting,
      'Designation':Designation,
      'Monthly_Income':MonthlyIncome
}])


if st.button("Predict Package"):
    prediction = model.predict(input_data)[0]
    result = "Product Selected" if prediction == 1 else "Product Not selected"
    st.subheader("Tourism Package Result:")
    st.success(f"The model predicts: **{result}**")
