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

def main():
    """
    Main execution flow for the CatBoost pipeline.
    """
    try:
        logger.info("Starting CatBoost pipeline execution")
        data_dir = Path('~/data/ml-zoomcamp-2024').expanduser()
        
        # Load data
        sales, stores, catalog = load_data(data_dir)
        
        # Preprocess data and compute monthly averages
        processed_data, top_100_combos, monthly_averages = preprocess_data(sales, stores, catalog)
        
        # Save preprocessed data for later use
        top_100_combos.to_csv(data_dir / 'top_100_combos.csv', index=False)
        monthly_averages.to_csv(data_dir / 'monthly_averages.csv', index=False)
        logger.info("Saved top 100 combinations and monthly averages to files")
        
        # Log fallback statistics
        fallback_count = top_100_combos['use_fallback'].sum()
        total_combos = len(top_100_combos)
        logger.info(f"Using fallback for {fallback_count}/{total_combos} combinations")
        logger.info("Initial data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
