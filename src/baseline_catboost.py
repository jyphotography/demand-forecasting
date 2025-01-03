import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the required datasets from the data directory.
    Returns sales, stores, and catalog dataframes.
    """
    try:
        logger.info("Loading data files...")
        sales = pd.read_csv(data_dir / 'sales.csv', index_col=0)
        stores = pd.read_csv(data_dir / 'stores.csv', index_col=0)
        catalog = pd.read_csv(data_dir / 'catalog.csv', index_col=0)
        
        # Drop any unnamed columns if present
        sales = sales.drop(columns=[col for col in sales.columns if 'Unnamed' in col])
        stores = stores.drop(columns=[col for col in stores.columns if 'Unnamed' in col])
        catalog = catalog.drop(columns=[col for col in catalog.columns if 'Unnamed' in col])
        
        return sales, stores, catalog
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def compute_monthly_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly averages for each store-item combination.
    
    Args:
        df: DataFrame containing the sales data with date and quantity columns
        
    Returns:
        DataFrame with monthly averages for each store-item combination
    """
    # Convert date to period for monthly grouping
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Compute monthly average for each store-item combination
    monthly_avg = (
        df.groupby(['store_id', 'item_id', 'year_month'])['quantity']
        .mean()
        .reset_index()
    )
    
    # Compute overall monthly average for each store-item combination
    overall_monthly_avg = (
        monthly_avg.groupby(['store_id', 'item_id'])['quantity']
        .mean()
        .reset_index()
        .rename(columns={'quantity': 'monthly_avg'})
    )
    
    return overall_monthly_avg

def needs_fallback(df: pd.DataFrame, threshold: int = 30) -> bool:
    """
    Determine if a store-item combination needs to use fallback monthly average.
    
    Args:
        df: DataFrame containing the sales data for a specific store-item combination
        threshold: Minimum number of data points required to avoid fallback
        
    Returns:
        bool: True if fallback should be used, False otherwise
    """
    return len(df) < threshold

def preprocess_data(sales: pd.DataFrame, stores: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess and merge the data, extract features, and limit to top 100 store-item combinations.
    Also computes monthly averages for fallback predictions.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            - processed_data: DataFrame containing the processed data for top 100 combinations
            - top_100_combos: DataFrame containing the selected store-item combinations
            - monthly_averages: DataFrame containing monthly averages for fallback predictions
    """
    logger.info("Starting data preprocessing...")
    
    # Convert IDs to strings
    sales['store_id'] = sales['store_id'].astype(str)
    sales['item_id'] = sales['item_id'].astype(str)
    stores['store_id'] = stores['store_id'].astype(str)
    catalog['item_id'] = catalog['item_id'].astype(str)
    
    # Merge data
    logger.info("Merging datasets...")
    df = sales.merge(stores, on='store_id', how='left')
    df = df.merge(catalog, on='item_id', how='left')
    
    # Convert date and extract features
    logger.info("Extracting time-based features...")
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Get top 100 store-item combinations by frequency
    logger.info("Selecting top 100 store-item combinations...")
    combo_counts = (
        df.groupby(['store_id', 'item_id'])
        .agg({'quantity': 'count'})
        .reset_index()
        .rename(columns={'quantity': 'count'})
    )
    top_100_combos = pd.DataFrame(
        combo_counts.nlargest(100, 'count')[['store_id', 'item_id']]
    ).reset_index(drop=True)
    
    # Filter data to top 100 combinations
    df = df.merge(top_100_combos, on=['store_id', 'item_id'], how='inner')
    
    # Compute monthly averages for fallback
    logger.info("Computing monthly averages for fallback predictions...")
    monthly_averages = compute_monthly_averages(df)
    
    # Mark combinations that need fallback
    combo_data = []
    for (store_id, item_id), group in df.groupby(['store_id', 'item_id']):
        needs_fb = needs_fallback(group)
        combo_data.append({
            'store_id': store_id,
            'item_id': item_id,
            'use_fallback': needs_fb,
            'data_points': len(group)
        })
    combo_status = pd.DataFrame(combo_data)
    
    # Update top_100_combos with fallback status
    top_100_combos = top_100_combos.merge(combo_status, on=['store_id', 'item_id'], how='left')
    
    logger.info(f"Preprocessed data shape: {df.shape}")
    logger.info(f"Number of unique store-item combinations: {len(top_100_combos)}")
    logger.info(f"Combinations using fallback: {top_100_combos['use_fallback'].sum()}")
    
    return df, top_100_combos, monthly_averages

def train_model(train_data: pd.DataFrame, cat_features: list) -> CatBoostRegressor:
    """
    Train a CatBoost model on the given data without validation.
    
    Args:
        train_data: DataFrame containing training data
        cat_features: List of categorical feature names
        
    Returns:
        Trained CatBoost model
    """
    logger.info("Preparing features for training...")
    
    # Prepare features and target
    feature_cols = [col for col in train_data.columns if col not in ['quantity', 'date', 'year_month']]
    X = train_data[feature_cols].copy()
    y = train_data['quantity']
    
    # Convert categorical features to string type
    for feat in cat_features:
        if feat in X.columns:
            X[feat] = X[feat].astype(str)
    
    # Initialize and train model with specified parameters
    logger.info("Training CatBoost model...")
    model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.1,
        l2_leaf_reg=3,
        cat_features=cat_features,
        verbose=False
    )
    
    model.fit(X, y)
    return model

def train_models(processed_data: pd.DataFrame, top_100_combos: pd.DataFrame) -> Dict[tuple[str, str], CatBoostRegressor]:
    """
    Train CatBoost models for store-item combinations with sufficient data.
    
    Args:
        processed_data: DataFrame containing the processed training data
        top_100_combos: DataFrame containing the top 100 store-item combinations with fallback status
        
    Returns:
        Dictionary mapping (store_id, item_id) to trained model
    """
    logger.info("Starting model training for combinations with sufficient data...")
    
    # Define categorical features
    cat_features = ['store_id', 'item_id', 'division', 'format', 'city', 'dept_name', 'class_name', 'subclass_name', 'item_type', 'month', 'day_of_week']
    
    # Initialize models dictionary
    models: Dict[tuple[str, str], CatBoostRegressor] = {}
    
    # Train models for combinations with sufficient data
    for _, row in top_100_combos[~top_100_combos['use_fallback']].iterrows():
        store_id, item_id = row['store_id'], row['item_id']
        
        # Get data for this combination
        combo_data = processed_data[
            (processed_data['store_id'] == store_id) & 
            (processed_data['item_id'] == item_id)
        ].copy()
        
        logger.info(f"Training model for store {store_id}, item {item_id}")
        try:
            model = train_model(combo_data, cat_features)
            models[(store_id, item_id)] = model
            logger.info(f"Successfully trained model for store {store_id}, item {item_id}")
        except Exception as e:
            logger.error(f"Failed to train model for store {store_id}, item {item_id}: {str(e)}")
            continue
    
    logger.info(f"Successfully trained {len(models)} models")
    return models

def process_test_data(test_df: pd.DataFrame, stores: pd.DataFrame, catalog: pd.DataFrame, top_100_combos: pd.DataFrame) -> pd.DataFrame:
    """
    Process test data consistently with training data preprocessing.
    
    Args:
        test_df: Raw test DataFrame
        stores: Stores DataFrame
        catalog: Catalog DataFrame
        top_100_combos: DataFrame containing the selected store-item combinations
        
    Returns:
        Processed test DataFrame with features ready for prediction
    """
    logger.info("Processing test data...")
    
    # Convert IDs and categorical columns to strings
    test_df['store_id'] = test_df['store_id'].astype(str)
    test_df['item_id'] = test_df['item_id'].astype(str)
    test_df['row_id'] = test_df.index.astype(str)
    
    # Convert categorical columns to strings
    cat_cols = ['division', 'format', 'city', 'dept_name', 'class_name', 'subclass_name', 'item_type']
    for col in cat_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str)
    stores['store_id'] = stores['store_id'].astype(str)
    catalog['item_id'] = catalog['item_id'].astype(str)
    
    # Convert row_id to string
    test_df['row_id'] = test_df['row_id'].astype(str)
    
    # Merge data
    df = test_df.merge(stores, on='store_id', how='left')
    df = df.merge(catalog, on='item_id', how='left')
    
    # Convert date and extract features
    # Convert date format from DD.MM.YYYY to YYYY-MM-DD
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Filter to top 100 combinations
    df = df.merge(top_100_combos[['store_id', 'item_id', 'use_fallback']], on=['store_id', 'item_id'], how='inner')
    
    logger.info(f"Processed test data shape: {df.shape}")
    return df

def generate_predictions(
    test_df: pd.DataFrame,
    models: Dict[tuple[str, str], CatBoostRegressor],
    monthly_averages: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate predictions for test data using trained models or monthly average fallback.
    
    Args:
        test_df: Processed test DataFrame
        models: Dictionary mapping (store_id, item_id) to trained model
        monthly_averages: DataFrame containing monthly averages for fallback
        
    Returns:
        DataFrame with row_id and quantity predictions
    """
    logger.info("Generating predictions...")
    predictions = []
    
    for _, row in test_df.iterrows():
        store_id, item_id = row['store_id'], row['item_id']
        row_id = row['row_id']
        
        if not row['use_fallback'] and (store_id, item_id) in models:
            # Use CatBoost model
            model = models[(store_id, item_id)]
            # Drop only columns that exist and aren't needed for prediction
            drop_cols = ['row_id', 'date']
            if 'quantity' in row.index:
                drop_cols.append('quantity')
            if 'use_fallback' in row.index:
                drop_cols.append('use_fallback')
            features = row.drop(drop_cols).to_frame().T
            pred = float(model.predict(features)[0])
        else:
            # Use monthly average fallback
            monthly_avg = monthly_averages[
                (monthly_averages['store_id'] == store_id) &
                (monthly_averages['item_id'] == item_id)
            ]['monthly_avg'].iloc[0]
            pred = float(monthly_avg)
        
        # Clip negative predictions to 0
        pred = max(0, pred)
        predictions.append({'row_id': row_id, 'quantity': pred})
    
    predictions_df = pd.DataFrame(predictions)
    logger.info(f"Generated {len(predictions_df)} predictions")
    return predictions_df

def main():
    """
    Main execution flow for the CatBoost pipeline.
    """
    try:
        logger.info("Starting CatBoost pipeline execution")
        # # data_dir = Path("/home/ubuntu/data/ml-zoomcamp-2024")
        data_dir = Path('/workspaces/demand-forecasting/data').expanduser()
        
        # Load data
        sales, stores, catalog = load_data(data_dir)
        
        # # Preprocess data and compute monthly averages
        processed_data, top_100_combos, monthly_averages = preprocess_data(sales, stores, catalog)
        
        # # Save preprocessed data for later use
        # top_100_combos.to_csv(data_dir / 'top_100_combos.csv', index=False)
        # monthly_averages.to_csv(data_dir / 'monthly_averages.csv', index=False)
        # logger.info("Saved top 100 combinations and monthly averages to files")
        
        # # Log fallback statistics
        # fallback_count = top_100_combos['use_fallback'].sum()
        # total_combos = len(top_100_combos)
        # logger.info(f"Using fallback for {fallback_count}/{total_combos} combinations")
        
        # Train models for combinations with sufficient data
        models = train_models(processed_data, top_100_combos)
        
        # # Create models directory
        # models_dir = data_dir / 'models'
        # models_dir.mkdir(exist_ok=True)
        
        # # Save trained models
        # logger.info("Saving trained models...")
        # for (store_id, item_id), model in models.items():
        #     model_path = models_dir / f'model_{store_id}_{item_id}.cbm'
        #     model.save_model(model_path)
        # logger.info(f"Saved {len(models)} models to {models_dir}")
        
        # Load and process test data
        logger.info("Loading test data...")
        test_df = pd.read_csv(data_dir / 'test.csv', sep=';', index_col=0)
        processed_test = process_test_data(test_df, stores, catalog, top_100_combos)
        
        # Generate predictions
        predictions_df = generate_predictions(processed_test, models, monthly_averages)
        
        # Save predictions
        predictions_df.to_csv(data_dir / 'catboost_submission.csv', index=False)
        logger.info("Saved predictions to catboost_submission.csv")
        
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
