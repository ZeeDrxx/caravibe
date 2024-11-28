import streamlit as st
from Modules.route_optimization import load_trip_data, optimize_routes
from Modules.demand_prediction import load_demand_data, train_model, predict_demand
from Modules.matching import load_user_data, preprocess_travel_history, train_similarity_model, find_similar_users


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


    # Preprocess travel data and train model
    user_data = load_user_data()
    tfidf_matrix, vectorizer = preprocess_travel_history(user_data)
    similarity_model = train_similarity_model(tfidf_matrix)
    
    # Add recommendation system to Streamlit app
    st.write("### Personalized Recommendations")
    user_id = st.selectbox("Select User ID for Recommendations", user_data['user_id'])
    gender_filter = st.checkbox("Apply Gender Filter", value=True)
    
    # Show options for new features
    st.write("#### Additional Filters")
    age_group_filter = st.selectbox("Select Age Group", user_data['age_group'].unique())
    travel_frequency_filter = st.selectbox("Select Income Level", user_data['travel_frequency'].unique())
    car_preference_filter = st.selectbox("Select Car Preference", user_data['car_preference'].unique())

    # Find similar users with the new filters
    recommendations = find_similar_users(user_id, user_data, similarity_model, tfidf_matrix, gender_filter)

    # Filter recommendations based on additional filters
    filtered_recommendations = {}
    for similar_user_id, details in recommendations.items():
        user_details = user_data[user_data['user_id'] == similar_user_id].iloc[0]
        
        # Apply additional feature filters
        if (user_details['age_group'] != age_group_filter or 
            user_details['travel_frequency'] != travel_frequency_filter or 
            user_details['car_preference'] != car_preference_filter):
            continue
        
        filtered_recommendations[similar_user_id] = details['travel_history']

    # Display recommendations
    if filtered_recommendations:
        st.write(f"Recommended Users for User {user_id}:")
        for similar_user_id, history in filtered_recommendations.items():
            st.write(f"- User {similar_user_id}: {history}")
    else:
        st.write("No suitable recommendations found.")

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
