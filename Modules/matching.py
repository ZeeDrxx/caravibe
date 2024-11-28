import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Configuration
from config import USER_DATA_PATH

def load_user_data():
    """
    Load user data from the CSV file.
    """
    try:
        return pd.read_csv(USER_DATA_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=["user_id", "first_name", "last_name", "gender", 
                                     "travel_with_only_females", "travel_history", 
                                     "age_group", "travel_frequency", "car_preference"])

def preprocess_travel_history(user_data):
    """
    Preprocess travel history to create a TF-IDF vectorized representation.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(user_data['travel_history'].fillna(""))
    return tfidf_matrix, vectorizer

def train_similarity_model(tfidf_matrix):
    """
    Train a K-Nearest Neighbors model on the travel history.
    """
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(tfidf_matrix)
    return model

def calculate_feature_similarity(user, other_user):
    """
    Calculate similarity score based on additional features.
    """
    # Encode age group to numerical value (higher is more similar)
    # Convert age groups to numerical values (e.g., using midpoints of ranges)
    age_group_mapping = {
        "18-24": 21, "25-34": 29, "35-44": 39, "45-54": 49,
        "55-64": 59, "56+": 70
    }
    # Convert age group to numerical values
    user_age = age_group_mapping.get(user['age_group'], 0)
    other_user_age = age_group_mapping.get(other_user['age_group'], 0)

    # Calculate age similarity
    age_similarity = 1 - abs(user_age - other_user_age) / 5
    travel_freq_mapping = {
        "Rarely": 1, "Monthly": 2, "Weekly": 3, "Daily": 4
    }
    # Convert travel frequency to numerical values
    user_travel_freq = travel_freq_mapping.get(user['travel_frequency'], 0)
    other_user_travel_freq = travel_freq_mapping.get(other_user['travel_frequency'], 0)
    # Calculate travel frequency similarity
    travel_freq_similarity = 1 - abs(user_travel_freq - other_user_travel_freq) / 3


    # Compare car preference (exact match is a full similarity)
    car_similarity = 1 if user['car_preference'] == other_user['car_preference'] else 0

    # Combine the individual feature similarities
    return (age_similarity + travel_freq_similarity + car_similarity) / 3

def find_similar_users(user_id, user_data, model, tfidf_matrix, gender_filter=True):
    # Get the user's index
    user_index = user_data.index[user_data['user_id'] == user_id].tolist()[0]

    # Find similar users based on travel history
    distances, indices = model.kneighbors(tfidf_matrix[user_index], n_neighbors=5)

    similar_users = {}
    for idx, distance in zip(indices[0], distances[0]):
        if idx == user_index or distance > 0.5:  # Exclude self and dissimilar users
            continue
        
        matched_user = user_data.iloc[idx]
        
        # Check gender constraints
        if gender_filter:
            if matched_user['gender'] == "Female" and matched_user['travel_with_only_females'] == "TRUE":
                continue
            if user_data.iloc[user_index]['gender'] == "Female" and user_data.iloc[user_index]['travel_with_only_females'] == "TRUE" and matched_user['gender'] != "Female":
                continue
        
        # Calculate feature similarity (age group, travel frequency, and car preference)
        feature_similarity = calculate_feature_similarity(user_data.iloc[user_index], matched_user)
        if feature_similarity < 0.6:  # Exclude users with low feature similarity
            continue

        # Add to recommendations if the feature similarity is high enough
        similar_users[matched_user['user_id']] = {
            "travel_history": matched_user['travel_history'],
            "similarity_score": feature_similarity
        }
    
    return similar_users
#income