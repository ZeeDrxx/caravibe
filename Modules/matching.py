import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
        return pd.DataFrame(columns=["user_id", "first_name", "last_name", "gender", "travel_with_only_females", "travel_history"])

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

def find_similar_users(user_id, user_data, model, tfidf_matrix, gender_filter=True):
    """
    Find similar users based on travel history and gender constraints.
    Args:
        user_id (int): The ID of the user for whom recommendations are generated.
        user_data (DataFrame): The full user dataset.
        model (NearestNeighbors): Trained KNN model.
        tfidf_matrix (sparse matrix): TF-IDF representation of travel history.
        gender_filter (bool): Whether to apply gender-based constraints.
    Returns:
        dict: Similar users and their travel histories.
    """
    # Get the user's index
    user_index = user_data.index[user_data['user_id'] == user_id].tolist()[0]

    # Find similar users
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
        
        # Add to recommendations
        similar_users[matched_user['user_id']] = matched_user['travel_history']
    
    return similar_users
