from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS module
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model (ensure the model is saved and in the same directory as app.py)
try:
    model = joblib.load('gym_population_density_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

# Define the list of days and times for one-hot encoding
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
times = ['Morning', 'Afternoon', 'Evening', 'Night']

# Helper function to align input features with model features
def align_features(input_df, model_features):
    for feature in model_features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Add missing feature columns with default value 0
    input_df = input_df[model_features]  # Reorder input_df columns to match model features
    return input_df

# Function to predict gym population density
@app.route('/predict', methods=['POST'])
def predict_density():
    try:
        data = request.get_json()

        # Extract required fields
        day = data['day']
        time = data['time']
        events_nearby = data['events_nearby']
        temperature = data['temperature']

        # Prepare input data for prediction (one-hot encode day and time)
        input_data = {
            'events_nearby': [events_nearby],
            'temperature': [temperature],
            **{f'day_{d}': [1 if d == day else 0] for d in days[1:]},  # One-hot encode day (skip Monday)
            **{f'time_{t}': [1 if t == time else 0] for t in times[1:]}  # One-hot encode time (skip Morning)
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data)

        # Align input features to match model features
        input_df_aligned = align_features(input_df, model.feature_names_in_)

        # Make the prediction
        predicted_density = model.predict(input_df_aligned)[0]

        # Return the prediction as a JSON response
        return jsonify({'predicted_population_density': predicted_density})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
