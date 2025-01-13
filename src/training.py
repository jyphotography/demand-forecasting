import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import pickle
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load and combine sales data"""
    logger.info("Loading data...")
    try:
        sales = pd.read_csv('../data/sales.csv', index_col=0)
        online = pd.read_csv('../data/online.csv', index_col=0)
        all_sales = pd.concat([sales, online], ignore_index=True)
        
        # Group by date, store_id, item_id and aggregate
        all_sales = all_sales.groupby(['date', 'store_id', 'item_id']).agg({
            'quantity': 'sum',
            'price_base': 'median'
        }).reset_index()
        
        return all_sales
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_features(df):
    """Prepare features for training"""
    logger.info("Preparing features...")
    try:
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Create time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Convert to string for categorical encoding
        df['date'] = df['date'].astype(str)
        df['store_id'] = df['store_id'].astype(str)
        df['year'] = df['year'].astype(str)
        df['month'] = df['month'].astype(str)
        df['day_of_week'] = df['day_of_week'].astype(str)
        
        # Handle missing values
        df = df.fillna('unknown')
        
        return df
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise

def train_random_forest(df_train, y_train):
    """Train Random Forest model with best parameters"""
    logger.info("Training Random Forest model...")
    try:
        # Convert to dictionary format
        dicts = df_train.to_dict(orient='records')
        
        # Initialize DictVectorizer
        dv = DictVectorizer(sparse=True)
        X_train = dv.fit_transform(dicts)
        
        # Initialize model with best parameters
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        return dv, model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def save_model(model, dv, output_path):
    """Save model and DictVectorizer to files"""
    logger.info("Saving model and vectorizer...")
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(output_dir / 'random_forest_model.bin', 'wb') as f_model:
            pickle.dump(model, f_model)
        
        # Save vectorizer
        with open(output_dir / 'dict_vectorizer.bin', 'wb') as f_dv:
            pickle.dump(dv, f_dv)
            
        logger.info(f"Model and vectorizer saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def main():
    """Main training pipeline"""
    try:
        # Load data
        all_sales = load_data()
        
        # # Filter data after November 2023
        # filtered_sales = all_sales[all_sales['date'] >= '2023-11-01']
        
        # Prepare features
        processed_sales = prepare_features(all_sales)
        
        # Split features and target
        X = processed_sales.drop('quantity', axis=1)
        y = processed_sales['quantity'].values
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        dv, model = train_random_forest(X_train, y_train)
        
        # Save model and vectorizer
        save_model(model, dv, '../models')
        
        # Validate model performance
        val_dicts = X_val.to_dict(orient='records')
        X_val_transformed = dv.transform(val_dicts)
        y_pred = model.predict(X_val_transformed)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        logger.info(f"Validation RMSE: {rmse:.4f}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()