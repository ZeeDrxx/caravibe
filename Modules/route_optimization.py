import pandas as pd

# Configuration
from config import TRIP_DATA_PATH

def load_trip_data():
    """
    Load trip data from the CSV file.
    """
    try:
        return pd.read_csv(TRIP_DATA_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=["start_location", "end_location", "distance_km", "duration_min"])

def optimize_routes(trip_data, max_distance_km=200):
    """
    Group trips based on proximity to optimize ride pooling.
    Args:
        trip_data (DataFrame): DataFrame containing trip data.
        max_distance_km (int): Maximum distance between grouped trips.
    Returns:
        DataFrame: Optimized trip groups.
    """
    trip_data['group_id'] = (trip_data['distance_km'] // max_distance_km).astype(int)
    grouped_trips = trip_data.groupby('group_id')
    
    optimized_routes = []
    for _, group in grouped_trips:
        group_summary = {
            "group_id": group['group_id'].iloc[0],
            "start_locations": ", ".join(group['start_location'].unique()),
            "end_locations": ", ".join(group['end_location'].unique()),
            "total_distance_km": group['distance_km'].sum(),
            "total_duration_min": group['duration_min'].sum(),
        }
        optimized_routes.append(group_summary)

    return pd.DataFrame(optimized_routes)
