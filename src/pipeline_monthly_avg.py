#!/usr/bin/env python3
"""
Monthly Average Prediction Pipeline for ML Zoomcamp 2024 Competition.
This script generates predictions based on average monthly quantity per item
from December 2023 to September 2024.
"""

import pandas as pd


def generate_submission():
    """
    Generate a submission file by computing monthly averages from historical data
    and applying them to test data. Uses data from 2023-12 to 2024-09 only.
    """
    # 1. Load historical sales
    print("Loading sales data...")
    sales = pd.read_csv('data/sales.csv')
    
    # Parse and filter data for specified date range
    sales['date'] = pd.to_datetime(sales['date'])
    mask = (sales['date'] >= '2023-12-01') & (sales['date'] <= '2024-09-30')
    sales_filtered = sales.loc[mask].copy()  # Create copy to avoid SettingWithCopyWarning
    
    print(f"Using {len(sales_filtered)} sales records from Dec 2023 to Sep 2024")
    
    # 2. Compute monthly averages
    # Convert to monthly period for grouping
    sales_filtered['year_month'] = sales_filtered['date'].dt.to_period('M')
    monthly_avgs = (
        sales_filtered.groupby(['item_id', 'store_id'])['quantity']
        .mean()
        .reset_index()
        .rename(columns={'quantity': 'monthly_avg'})
    )
    
    print(f"Computed averages for {len(monthly_avgs)} item-store combinations")
    
    # 3. Read the test CSV and prepare for predictions
    print("Loading test data...")
    # Read test data with semicolon separator and split columns
    test = pd.read_csv('data/test.csv', sep=';')
    
    # Convert date format (DD.MM.YYYY to YYYY-MM-DD)
    test['date'] = pd.to_datetime(test['date'], format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
    test['quantity'] = 0.0  # default value for missing combinations
    
    # Merge test with monthly averages
    merged = test.merge(monthly_avgs, on=['item_id', 'store_id'], how='left')
    merged['quantity'] = merged['monthly_avg'].fillna(0.0)
    
    # 4. Format and save submission
    submission = merged[['row_id', 'quantity']]
    submission.to_csv('submission.csv', index=False)
    
    print(f"Generated predictions for {len(submission)} test entries")
    print("Submission saved to submission.csv")


if __name__ == "__main__":
    generate_submission()
