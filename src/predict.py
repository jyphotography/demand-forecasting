import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
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

def prepare_test_data(test_df):
    """Prepare test data for prediction"""
    try:
        # Convert date to datetime
        test_df['date'] = pd.to_datetime(test_df['date'], format='%d.%m.%Y')
        
        # Create time features
        test_df['year'] = test_df['date'].dt.year
        test_df['month'] = test_df['date'].dt.month
        test_df['day_of_week'] = test_df['date'].dt.dayofweek
        
        # Convert to string
        test_df['date'] = test_df['date'].astype(str)
        test_df['store_id'] = test_df['store_id'].astype(str)
        test_df['year'] = test_df['year'].astype(str)
        test_df['month'] = test_df['month'].astype(str)
        test_df['day_of_week'] = test_df['day_of_week'].astype(str)
        
        return test_df
    except Exception as e:
        logger.error(f"Error preparing test data: {str(e)}")
        raise

def make_predictions(test_df, model, dv):
    """Make predictions using loaded model"""
    try:
        dicts = test_df.to_dict(orient='records')
        X = dv.transform(dicts)
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise

def main():
    """Main prediction pipeline"""
    try:
        # Load model and vectorizer
        model, dv = load_model('../models')
        
        # Load and prepare test data
        test = pd.read_csv('../data/test.csv', sep=';')
        processed_test = prepare_test_data(test)
        
        # Make predictions
        predictions = make_predictions(processed_test, model, dv)
        
        # Create submission file
        submission = pd.DataFrame({
            'row_id': test.index,
            'quantity': predictions
        })
        
        # Save predictions
        submission.to_csv('../submissions/rf_predictions.csv', index=False)
        logger.info("Predictions saved to submissions/rf_predictions.csv")
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()