import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from pathlib import Path
from datetime import datetime
import logging

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
        sales = pd.read_csv(data_dir / 'sales.csv')
        stores = pd.read_csv(data_dir / 'stores.csv')
        catalog = pd.read_csv(data_dir / 'catalog.csv')
        
        # Drop any unnamed columns if present
        sales = sales.drop(columns=[col for col in sales.columns if 'Unnamed' in col])
        stores = stores.drop(columns=[col for col in stores.columns if 'Unnamed' in col])
        catalog = catalog.drop(columns=[col for col in catalog.columns if 'Unnamed' in col])
        
        return sales, stores, catalog
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(sales: pd.DataFrame, stores: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess and merge the data, extract features, and limit to top 100 store-item combinations.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (processed_data, top_100_combos)
            - processed_data: DataFrame containing the processed data for top 100 combinations
            - top_100_combos: DataFrame containing the selected store-item combinations
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
    
    logger.info(f"Preprocessed data shape: {df.shape}")
    logger.info(f"Number of unique store-item combinations: {len(top_100_combos)}")
    
    return df, top_100_combos

def main():
    """
    Main execution flow for the CatBoost pipeline.
    """
    try:
        logger.info("Starting CatBoost pipeline execution")
        data_dir = Path('~/data/ml-zoomcamp-2024').expanduser()
        
        # Load data
        sales, stores, catalog = load_data(data_dir)
        
        # Preprocess data
        processed_data, top_100_combos = preprocess_data(sales, stores, catalog)
        
        # Save top 100 combinations for later use with test data
        top_100_combos.to_csv(data_dir / 'top_100_combos.csv', index=False)
        logger.info("Saved top 100 combinations to file")
        
        logger.info("Initial data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
