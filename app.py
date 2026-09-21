import streamlit as st
import joblib
import pandas as pd

# Load the trained model
# Make sure 'delivery_delay.sav' is in the same directory as this Streamlit app or provide the full path
model = joblib.load('delivery_delay.sav')

# Define the columns that the model expects, based on X.columns from your training data
# This order is crucial for correct predictions
model_columns = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                 'Warehouse_Processing_Time']

# --- Streamlit App Layout ---
st.set_page_config(page_title="Delivery Delay Prediction", layout="centered")

st.title("🚚 Delivery Delay Prediction App")
st.markdown("Enter the details below to predict if a delivery will be delayed.")

st.header("Input Features")

# Create input fields for each feature
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delivery_distance = st.number_input("Delivery Distance (km)", min_value=0.0, value=20.0, step=0.1)
        traffic_congestion = st.selectbox("Traffic Congestion (1-5)", options=[1, 2, 3, 4, 5], index=2)
        weather_condition = st.selectbox("Weather Condition (1-5)", options=[1, 2, 3, 4, 5], index=2)
        delivery_slot = st.selectbox("Delivery Slot (1-3)", options=[1, 2, 3], index=1)
    
    with col2:
        driver_experience = st.number_input("Driver Experience (years)", min_value=0, value=5, step=1)
        num_stops = st.number_input("Number of Stops", min_value=1, value=5, step=1)
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, value=3, step=1)
        road_condition_score = st.selectbox("Road Condition Score (1-5)", options=[1, 2, 3, 4, 5], index=2)
    
    with col3:
        package_weight = st.number_input("Package Weight (kg)", min_value=0.0, value=10.0, step=0.1)
        fuel_efficiency = st.number_input("Fuel Efficiency (km/L)", min_value=0.0, value=10.0, step=0.1)
        warehouse_processing_time = st.number_input("Warehouse Processing Time (mins)", min_value=0, value=45, step=1)

# Make prediction button
if st.button("Predict Delivery Delay"):
    # Create a DataFrame from the input values, ensuring correct order and names
    input_data = pd.DataFrame([[delivery_distance, traffic_congestion, weather_condition,
                                delivery_slot, driver_experience, num_stops, vehicle_age,
                                road_condition_score, package_weight, fuel_efficiency,
                                warehouse_processing_time]],
                              columns=model_columns)
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Display results
    st.subheader("Prediction Results:")
    if prediction[0] == 1:
        st.error("The model predicts: **DELIVERY WILL LIKELY BE DELAYED**")
    else:
        st.success("The model predicts: **DELIVERY IS LIKELY ON TIME**")
        
    st.write(f"Probability of Delay: {prediction_proba[0][1]*100:.2f}%")
    st.write(f"Probability of On-Time: {prediction_proba[0][0]*100:.2f}%")

# Instructions on how to run this app (for clarity outside the app itself)
st.markdown("--- --- ")
st.markdown("**To run this Streamlit app locally:**")
st.markdown("1. Save the code above as `streamlit_app.py` in the same directory as your `delivery_delay.sav` model file.")
st.markdown("2. Open your terminal or command prompt.")
st.markdown("3. Navigate to the directory where you saved the files.")
st.markdown("4. Run the command: `streamlit run streamlit_app.py`")
st.markdown("5. Your browser should automatically open to the Streamlit app.")
