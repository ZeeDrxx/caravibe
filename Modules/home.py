import streamlit as st
from Modules.route_optimization import load_trip_data, optimize_routes
from Modules.demand_prediction import load_demand_data, train_model, predict_demand

def home_page():
    # Load trip data and user
    trip_data = load_trip_data()

    st.title("CaraVibe: AI-Powered Ridesharing App")


    st.subheader("Dynamic Route Optimization")
    # Display original trip data
    st.write("### Original Trip Data")
    st.dataframe(trip_data)
    # Show the original trip data
    st.write("### Original Trip Data")
    st.dataframe(trip_data)
    # Optimize routes
    st.write("### Optimized Routes")
    max_distance = st.slider("Select Max Distance for Optimization (km)", 50, 500, 200)
    optimized_routes = optimize_routes(trip_data, max_distance_km=max_distance)
    st.dataframe(optimized_routes)
    # Provide a download option for optimized routes
    st.download_button(
        label="Download Optimized Routes",
        data=optimized_routes.to_csv(index=False),
        file_name="optimized_routes.csv",
        mime="text/csv",
    )


    # Load demand data and extract unique locations
    demand_data, unique_start_locations, unique_end_locations = load_demand_data()
    # Demand Prediction Section
    st.write("### Demand Prediction")
    user_day_of_week = st.selectbox("Select Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    user_start_location = st.selectbox("Select Start Location", unique_start_locations)
    user_end_location = st.selectbox("Select End Location", unique_end_locations)
    user_time_of_day = st.selectbox("Select Time of Day", ["morning", "afternoon", "evening"])
    # Train model (if needed)
    model, label_encoders = train_model(demand_data)
    # Predict the demand based on user input
    prediction = predict_demand(user_day_of_week, user_start_location, user_end_location, user_time_of_day, model, label_encoders)
    # Display the predicted demand
    st.write(f"Predicted Demand: {prediction['predicted_demand']}")