import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_data(data_dir, sample_size=None):
    """Load all required datasets
    Args:
        data_dir: Path to data directory
        sample_size: Number of rows to sample from sales/online data for development
    """
    print(f"Loading data with sample_size: {sample_size if sample_size else 'Full'}")
    
    # Load sales data with optional sampling
    sales = pd.read_csv(data_dir / 'sales.csv', nrows=sample_size)
    online = pd.read_csv(data_dir / 'online.csv', nrows=sample_size)
    
    # Get unique item_ids and store_ids from sampled data
    item_ids = pd.concat([sales['item_id'], online['item_id']]).unique()
    store_ids = pd.concat([sales['store_id'], online['store_id']]).unique()
    
    # Load catalog and filter for relevant items
    catalog = pd.read_csv(data_dir / 'catalog.csv')
    if sample_size:
        catalog = catalog[catalog['item_id'].isin(item_ids)]
    
    # Load stores and filter for relevant stores
    stores = pd.read_csv(data_dir / 'stores.csv')
    if sample_size:
        stores = stores[stores['store_id'].isin(store_ids)]
    
    # Load test data
    test = pd.read_csv(data_dir / 'test.csv', sep=';')
    
    return sales, online, catalog, stores, test

def preprocess_data(sales, online, catalog, stores):
    """Basic preprocessing of the data"""
    # Combine sales and online
    sales['channel'] = 'offline'
    online['channel'] = 'online'
    all_sales = pd.concat([sales, online], ignore_index=True)
    
    # Convert date to datetime
    all_sales['date'] = pd.to_datetime(all_sales['date'])
    
    # Create basic time features
    all_sales['year'] = all_sales['date'].dt.year
    all_sales['month'] = all_sales['date'].dt.month
    all_sales['day'] = all_sales['date'].dt.day
    all_sales['day_of_week'] = all_sales['date'].dt.dayofweek
    
    # Fill missing values in catalog
    catalog['dept_name'] = catalog['dept_name'].fillna('Unknown')
    catalog['class_name'] = catalog['class_name'].fillna('Unknown')
    
    # Merge with catalog and stores
    all_sales = all_sales.merge(catalog[['item_id', 'dept_name', 'class_name']], 
                               on='item_id', how='left')
    all_sales = all_sales.merge(stores[['store_id', 'format', 'city']], 
                               on='store_id', how='left')
    
    return all_sales

def prepare_features(df, train_data=None, is_train=True):
    """Prepare features for model training or prediction
    Args:
        df: DataFrame to prepare
        train_data: Training data for computing aggregated features (for test data)
        is_train: Whether this is training data (has channel column) or test data
    """
    # Encode categorical variables
    categorical_cols = ['dept_name', 'class_name', 'format', 'city']
    if is_train:
        categorical_cols.append('channel')
    
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    # Create or apply aggregated features
    if is_train or train_data is not None:
        source_data = train_data if train_data is not None else df
        agg_features = source_data.groupby(['item_id', 'store_id']).agg({
            'quantity': ['mean', 'std', 'max'],
            'price_base': ['mean', 'std']
        }).reset_index()
        
        agg_features.columns = ['item_id', 'store_id', 
                               'item_store_qty_mean', 'item_store_qty_std', 'item_store_qty_max',
                               'item_store_price_mean', 'item_store_price_std']
        
        df = df.merge(agg_features, on=['item_id', 'store_id'], how='left')
        
        # Fill missing values in aggregated features
        df['item_store_qty_std'] = df['item_store_qty_std'].fillna(0)
        df['item_store_price_std'] = df['item_store_price_std'].fillna(0)
        df['item_store_qty_mean'] = df['item_store_qty_mean'].fillna(df['item_store_qty_mean'].mean())
        df['item_store_qty_max'] = df['item_store_qty_max'].fillna(df['item_store_qty_max'].mean())
        df['item_store_price_mean'] = df['item_store_price_mean'].fillna(df['item_store_price_mean'].mean())
    
    return df

def validate_preprocessing(df):
    """Validate preprocessing steps"""
    print("\nValidating preprocessing...")
    print(f"Total rows: {len(df)}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nFeature info:")
    print(df.info())
    print("\nSample of processed data:")
    print(df.head())
    return df

def train_model(train_data, features, target='quantity'):
    """Train LightGBM model"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    dtrain = lgb.Dataset(train_data[features], train_data[target])
    dval = lgb.Dataset(val_data[features], val_data[target], reference=dtrain)
    
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        callbacks=callbacks
    )
    
    return model

def prepare_test_data(test_df, catalog, stores):
    """Prepare test data for prediction"""
    test_df['date'] = pd.to_datetime(test_df['date'], format='%d.%m.%Y')
    
    # Create time features
    test_df['year'] = test_df['date'].dt.year
    test_df['month'] = test_df['date'].dt.month
    test_df['day'] = test_df['date'].dt.day
    test_df['day_of_week'] = test_df['date'].dt.dayofweek
    
    # Merge with catalog and stores
    test_df = test_df.merge(catalog[['item_id', 'dept_name', 'class_name']], 
                           on='item_id', how='left')
    test_df = test_df.merge(stores[['store_id', 'format', 'city']], 
                           on='store_id', how='left')
    
    return test_df

def generate_submission(model, test_df, features, submission_file):
    """Generate submission file"""
    predictions = model.predict(test_df[features])
    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'quantity': predictions
    })
    submission.to_csv(submission_file, index=False)
    return submission

def main():
    # Set up paths
    data_dir = Path('~/data/ml-zoomcamp-2024').expanduser()
    submission_file = data_dir / 'submission.csv'
    
    # Load data with sampling for development
    print("Loading data...")
    sample_size = 10000  # Use smaller sample for development
    sales, online, catalog, stores, test = load_data(data_dir, sample_size=sample_size)
    
    # Preprocess training data
    print("Preprocessing training data...")
    train_data = preprocess_data(sales, online, catalog, stores)
    train_data = prepare_features(train_data, is_train=True)
    
    # Validate preprocessing
    train_data = validate_preprocessing(train_data)
    
    if input("Continue with model training? (y/n): ").lower() != 'y':
        return
        
    # Define features
    features = ['year', 'month', 'day', 'day_of_week',
                'item_store_qty_mean', 'item_store_qty_std', 'item_store_qty_max',
                'item_store_price_mean', 'item_store_price_std',
                'dept_name', 'class_name', 'format', 'city']
    
    # Train model
    print("Training model...")
    model = train_model(train_data, features)
    
    # Prepare test data
    print("Preparing test data...")
    test_data = prepare_test_data(test, catalog, stores)
    test_data = prepare_features(test_data, train_data=train_data, is_train=False)
    
    # Generate submission
    print("Generating submission...")
    submission = generate_submission(model, test_data, features, submission_file)
    print(f"Submission saved to {submission_file}")

if __name__ == "__main__":
    main()
