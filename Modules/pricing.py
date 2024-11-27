# modules/pricing.py

import streamlit as st
from config import BASE_FARE, PER_KM_FARE

def dynamic_pricing():
    st.write("### Dynamic Pricing")
    st.write("Your fare is dynamically adjusted based on real-time demand and distance.")

    distance = st.slider("Select trip distance (in km)", 1, 100, 10)  # Default 10 km
    demand_factor = st.slider("Select demand factor", 1, 3, 1)  # Demand multiplier (1-3)

    total_fare = BASE_FARE + (PER_KM_FARE * distance * demand_factor)
    st.write(f"Total Fare: {total_fare} MAD")
