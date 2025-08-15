import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, encoder
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')
zipcode_encoder = joblib.load('zipcode_encoder.pkl')

st.title("🏠 House Price Prediction App")

# User Inputs
bedrooms = st.number_input('Bedrooms', min_value=0, max_value=20, value=3)
bathrooms = st.number_input('Bathrooms', min_value=0.0, max_value=10.0, value=2.0)
sqft_living = st.number_input('Sqft Living', min_value=0, max_value=20000, value=1500)
sqft_lot = st.number_input('Sqft Lot', min_value=0, max_value=100000, value=5000)
floors = st.number_input('Floors', min_value=1.0, max_value=4.0, value=1.0)
waterfront = st.selectbox('Waterfront (0=No, 1=Yes)', options=[0,1], index=0)
view = st.slider('View (0-4)', 0, 4, 0)
condition = st.slider('Condition (1-5)', 1, 5, 3)
grade = st.slider('Grade (1-13)', 1, 13, 7)
sqft_above = st.number_input('Sqft Above', min_value=0, max_value=15000, value=1200)
sqft_basement = st.number_input('Sqft Basement', min_value=0, max_value=10000, value=300)
zipcode = st.number_input('Zipcode', min_value=98000, max_value=98199, value=98001)
lat = st.number_input('Latitude', min_value=47.0, max_value=48.0, value=47.5)
long = st.number_input('Longitude', min_value=-122.5, max_value=-121.5, value=-122.2)
sqft_living15 = st.number_input('Sqft Living15', min_value=0, max_value=20000, value=1500)
sqft_lot15 = st.number_input('Sqft Lot15', min_value=0, max_value=100000, value=5000)
yr_built = st.number_input('Year Built', min_value=1800, max_value=2025, value=1990)
yr_renovated = st.number_input('Year Renovated', min_value=0, max_value=2025, value=0)

# Create DataFrame
input_df = pd.DataFrame([{
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Sqft_living': sqft_living,
    'Sqft_lot': sqft_lot,
    'Floors': floors,
    'Waterfront': waterfront,
    'View': view,
    'Condition': condition,
    'Grade': grade,
    'Sqft_above': sqft_above,
    'Sqft_basement': sqft_basement,
    'zipcode': zipcode,
    'Lat': lat,
    'Long': long,
    'Sqft_living15': sqft_living15,
    'Sqft_lot15': sqft_lot15
}])

# Feature Engineering
input_df['House_Age'] = 2025 - yr_built
input_df['Was_Renovated'] = 0 if yr_renovated == 0 else 1
input_df['Total_Bathrooms'] = bathrooms + 0.5 * floors
input_df['Living_per_Lot'] = sqft_living / sqft_lot if sqft_lot != 0 else 0
input_df['Total_Sqft'] = sqft_living + sqft_basement

# Encode zipcode
input_df['zipcode'] = zipcode_encoder.transform(input_df[['zipcode']])

# Reorder columns to match training
final_features = ['Bedrooms', 'Bathrooms', 'Sqft_living', 'Sqft_lot', 'Floors', 'Waterfront', 'View',
                  'Condition', 'Grade', 'Sqft_above', 'Sqft_basement', 'zipcode', 'Lat', 'Long',
                  'Sqft_living15', 'Sqft_lot15', 'House_Age', 'Was_Renovated',
                  'Total_Bathrooms', 'Living_per_Lot', 'Total_Sqft']

input_df = input_df[final_features]

# Scale
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.success(f"🏡 Estimated House Price: ${prediction[0]:,.2f}")
