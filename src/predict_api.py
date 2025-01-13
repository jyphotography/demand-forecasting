from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def load_model(model_path='../models'):
    """Load saved model and vectorizer"""
    try:
        with open(Path(model_path) / 'random_forest_model.bin', 'rb') as f_model:
            model = pickle.load(f_model)
        with open(Path(model_path) / 'dict_vectorizer.bin', 'rb') as f_dv:
            dv = pickle.load(f_dv)
        return model, dv
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def prepare_features(item_data):
    """Prepare features for a single item prediction"""
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([item_data])
        
        # Convert date to datetime (using DD.MM.YYYY format)
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
        
        # Create time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Convert to string for categorical encoding
        df['date'] = df['date'].astype(str)
        df['store_id'] = df['store_id'].astype(str)
        df['item_id'] = df['item_id'].astype(str)
        df['year'] = df['year'].astype(str)
        df['month'] = df['month'].astype(str)
        df['day_of_week'] = df['day_of_week'].astype(str)
        
        # Add price if available, otherwise set to None
        if 'price' not in df.columns:
            df['price'] = None
            
        return df
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise

# Load model and vectorizer at startup
model, dv = load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get input data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['item_id', 'store_id', 'date']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate date format (DD.MM.YYYY)
        try:
            datetime.strptime(data['date'], '%d.%m.%Y')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use DD.MM.YYYY'}), 400
            
        # Prepare features
        df = prepare_features(data)
        
        # Transform data and make prediction
        dicts = df.to_dict(orient='records')
        X = dv.transform(dicts)
        prediction = model.predict(X)
        
        # Round prediction to nearest integer (since quantity should be whole number)
        prediction = round(float(prediction[0]))
        
        # Prepare response
        response = {
            # 'item_id': data['item_id'],
            # 'store_id': data['store_id'],
            # 'date': data['date'],
            'predicted_quantity': prediction
        }
        
        logger.info(f"Prediction made for item_id: {data['item_id']}, store_id: {data['store_id']}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=True) 