import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow import keras

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load Artifacts
print("Loading model and preprocessor...")
try:
    with open(os.path.join(MODELS_DIR, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)
    
    model = keras.models.load_model(os.path.join(MODELS_DIR, 'yield_model.keras'))
    print("Artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    raise e

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Create DataFrame from input (expecting a dictionary or list of dictionaries)
        # Note: The keys must match the features used during training:
        # 'tuition_in_state', 'sat_avg', 'pell_grant_rate', 'faculty_salary', 'state', 'control'
        
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
            
        # Preprocess
        # The preprocessor expects specific column names. 
        # Ensure input data keys match training feature names.
        
        # Check for missing columns and fill with NaNs if necessary (impuer will handle it)
        # However, for a robust API, we should probably validate input. 
        # For this demo, we assume valid input keys.
        
        X_processed = preprocessor.transform(df)
        
        # Convert to dense if sparse
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
            
        # Predict
        predictions = model.predict(X_processed)
        
        # Convert predictions towards readable format
        results = []
        for i, pred in enumerate(predictions):
            yield_score = float(pred[0])
            results.append({
                'predicted_yield': yield_score,
                'yield_category': 'High' if yield_score > 0.5 else 'Low' # Simplified threshold
            })
            
        return jsonify({'predictions': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("Starting Flask server...")
    print("\nSAMPLE REQUEST JSON:")
    print("""
    {
        "school_name": "Test University",
        "state": "CA",
        "control": 1,
        "tuition_in_state": 7000,
        "sat_avg": 1200,
        "pell_grant_rate": 0.45,
        "faculty_salary": 9000
    }
    """)
    app.run(host='0.0.0.0', port=50001)
    
"""
SAMPLE JSON for POST /predict:

{
    "school_name": "Future Tech University",
    "state": "MA",
    "control": 2,
    "tuition_in_state": 55000,
    "sat_avg": 1450,
    "pell_grant_rate": 0.15,
    "faculty_salary": 12000
}

Note: 'control' 1=Public, 2=Private
"""
