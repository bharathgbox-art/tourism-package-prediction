"""
app.py  –  Streamlit Deployment
"""

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🌿",
    layout="wide",
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_model.pkl",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Exact feature order the model was trained with
FEATURE_COLUMNS = [
    "Unnamed: 0", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

CONTACT_MAP     = {"Company Invited": 0, "Self Inquiry": 1}
OCCUPATION_MAP  = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
GENDER_MAP      = {"Female": 0, "Male": 1}
MARITAL_MAP     = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
DESIGNATION_MAP = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}
PRODUCT_MAP     = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}

st.title("🌿 Wellness Tourism Package – Purchase Predictor")
st.markdown(
    "Enter the customer details in the sidebar, then click **Predict** to see "
    "whether the customer is likely to purchase the Wellness Tourism Package."
)

with st.sidebar:
    st.header("Customer Details")
    age                     = st.slider("Age", 18, 80, 35)
    type_of_contact         = st.selectbox("Type of Contact", list(CONTACT_MAP.keys()))
    city_tier               = st.selectbox("City Tier", [1, 2, 3])
    occupation              = st.selectbox("Occupation", list(OCCUPATION_MAP.keys()))
    gender                  = st.selectbox("Gender", list(GENDER_MAP.keys()))
    marital_status          = st.selectbox("Marital Status", list(MARITAL_MAP.keys()))
    designation             = st.selectbox("Designation", list(DESIGNATION_MAP.keys()))
    monthly_income          = st.number_input("Monthly Income (₹)", 10000, 100000, 30000, step=1000)
    passport                = st.selectbox("Has Passport?", ["No", "Yes"])
    own_car                 = st.selectbox("Owns a Car?", ["No", "Yes"])

    st.header("Trip Details")
    num_persons_visiting    = st.slider("Number of Persons Visiting", 1, 5, 2)
    preferred_property_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
    num_trips               = st.slider("Number of Trips Per Year", 1, 22, 3)
    num_children_visiting   = st.slider("Number of Children Visiting (< 5 yrs)", 0, 3, 0)

    st.header("Sales Interaction")
    pitch_satisfaction      = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched         = st.selectbox("Product Pitched", list(PRODUCT_MAP.keys()))
    num_followups           = st.slider("Number of Follow-ups", 1, 6, 3)
    duration_of_pitch       = st.slider("Duration of Pitch (minutes)", 5, 60, 15)

    predict_btn = st.button("Predict", use_container_width=True)

if predict_btn:
    # Build row matching the exact feature order the model was trained with
    # 'Unnamed: 0' was the dataframe index accidentally saved — set to 0
    row = {
        "Unnamed: 0":               0,
        "Age":                      age,
        "TypeofContact":            CONTACT_MAP[type_of_contact],
        "CityTier":                 city_tier,
        "DurationOfPitch":          duration_of_pitch,
        "Occupation":               OCCUPATION_MAP[occupation],
        "Gender":                   GENDER_MAP[gender],
        "NumberOfPersonVisiting":   num_persons_visiting,
        "NumberOfFollowups":        num_followups,
        "ProductPitched":           PRODUCT_MAP[product_pitched],
        "PreferredPropertyStar":    preferred_property_star,
        "MaritalStatus":            MARITAL_MAP[marital_status],
        "NumberOfTrips":            num_trips,
        "Passport":                 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore":   pitch_satisfaction,
        "OwnCar":                   1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children_visiting,
        "Designation":              DESIGNATION_MAP[designation],
        "MonthlyIncome":            monthly_income,
    }

    input_data = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("✅ **Likely to Purchase** the Wellness Package")
        else:
            st.error("❌ **Unlikely to Purchase** the Wellness Package")

    with col2:
        st.metric(label="Purchase Probability", value=f"{probability:.1%}")

    st.subheader("Input Summary")
    display_cols = [c for c in input_data.columns if c != "Unnamed: 0"]
    st.dataframe(input_data[display_cols], use_container_width=True)
