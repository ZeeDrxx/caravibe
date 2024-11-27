import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import DEMAND_DATA_PATH

def load_demand_data():
    try:
        # Load data from the file path specified in the config
        demand_data = pd.read_csv(DEMAND_DATA_PATH)
        
        # Check if required columns exist
        required_columns = {'day_of_week', 'start_location', 'end_location', 'time_of_day', 'demand'}
        if not required_columns.issubset(demand_data.columns):
            raise ValueError(f"Missing one or more required columns: {required_columns}")
        
        # Extract unique start and end locations
        unique_start_locations = demand_data['start_location'].unique().tolist()
        unique_end_locations = demand_data['end_location'].unique().tolist()
        
        return demand_data, unique_start_locations, unique_end_locations
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The demand data file was not found at {DEMAND_DATA_PATH}. Please check the file path.")
    except pd.errors.EmptyDataError:
        raise ValueError("The demand data file is empty.")
    except Exception as e:
        raise RuntimeError(f"Error loading demand data: {e}")


def train_model(demand_data):
    """
    Train a machine learning model to predict demand based on historical data.
    """
    # Encode categorical columns
    label_encoders = {}
    for column in ['day_of_week', 'start_location', 'end_location', 'time_of_day']:
        encoder = LabelEncoder()
        demand_data[column] = encoder.fit_transform(demand_data[column])
        label_encoders[column] = encoder

    # Split data into features and target
    X = demand_data[['day_of_week', 'start_location', 'end_location', 'time_of_day']]
    y = demand_data['demand']

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Return the trained model and label encoders
    return model, label_encoders


def predict_demand(day_of_week, start_location, end_location, time_of_day, model, label_encoders):
    try:
        # Transform inputs using the encoders
        encoded_input = {
            "day_of_week": label_encoders['day_of_week'].transform([day_of_week])[0],
            "start_location": label_encoders['start_location'].transform([start_location])[0],
            "end_location": label_encoders['end_location'].transform([end_location])[0],
            "time_of_day": label_encoders['time_of_day'].transform([time_of_day])[0],
        }

        # Convert the encoded input into a DataFrame with the correct feature names
        input_df = pd.DataFrame([encoded_input])

        # Predict demand and round to the nearest integer
        predicted_demand = model.predict(input_df)[0]
        rounded_demand = int(round(predicted_demand))

        return {
            "day_of_week": day_of_week,
            "start_location": start_location,
            "end_location": end_location,
            "time_of_day": time_of_day,
            "predicted_demand": rounded_demand,
        }
    except Exception as e:
        return {
            "day_of_week": day_of_week,
            "start_location": start_location,
            "end_location": end_location,
            "time_of_day": time_of_day,
            "predicted_demand": f"Prediction failed: {str(e)}",
        }
    #
#