"""
app.py  –  Streamlit Deployment
--------------------------------
Loads the trained model from the repository, collects customer inputs,
assembles them into a DataFrame with the exact same column names and
order used during training, and displays a purchase prediction.
"""

import os
import joblib
import pandas as pd
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🌿",
    layout="wide",
)

# ── Load the model ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_model.pkl",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Feature names in exact training order (from prep.py after dropping CustomerID & ProdTaken) ──
FEATURE_COLUMNS = [
    "Age", "TypeofContact", "CityTier", "Occupation", "Gender",
    "NumberOfPersonVisiting", "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome", "PitchSatisfactionScore",
    "ProductPitched", "NumberOfFollowups", "DurationOfPitch",
]

# ── Encoding maps (must match LabelEncoder alphabetical order used in prep.py) ──
CONTACT_MAP     = {"Company Invited": 0, "Self Inquiry": 1}
OCCUPATION_MAP  = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
GENDER_MAP      = {"Female": 0, "Male": 1}
MARITAL_MAP     = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
DESIGNATION_MAP = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}
PRODUCT_MAP     = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}

# ── UI ────────────────────────────────────────────────────────────────────
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

# ── Prediction ────────────────────────────────────────────────────────────
if predict_btn:
    # Build a dict in the exact same column order used during training
    row = {
        "Age":                      age,
        "TypeofContact":            CONTACT_MAP[type_of_contact],
        "CityTier":                 city_tier,
        "Occupation":               OCCUPATION_MAP[occupation],
        "Gender":                   GENDER_MAP[gender],
        "NumberOfPersonVisiting":   num_persons_visiting,
        "PreferredPropertyStar":    preferred_property_star,
        "MaritalStatus":            MARITAL_MAP[marital_status],
        "NumberOfTrips":            num_trips,
        "Passport":                 1 if passport == "Yes" else 0,
        "OwnCar":                   1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children_visiting,
        "Designation":              DESIGNATION_MAP[designation],
        "MonthlyIncome":            monthly_income,
        "PitchSatisfactionScore":   pitch_satisfaction,
        "ProductPitched":           PRODUCT_MAP[product_pitched],
        "NumberOfFollowups":        num_followups,
        "DurationOfPitch":          duration_of_pitch,
    }

    # Create DataFrame with columns in exact training order
    input_data = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    # Show what the model expects vs what we are sending
    try:
        expected = model.feature_names_in_.tolist()
        if expected != FEATURE_COLUMNS:
            st.warning(
                f"Feature mismatch detected!\n"
                f"Model expects: {expected}\n"
                f"App sending:   {FEATURE_COLUMNS}"
            )
            # Re-order to match model exactly
            input_data = input_data[expected]
    except AttributeError:
        pass  # Older sklearn versions don't have feature_names_in_

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
    st.dataframe(input_data, use_container_width=True)
