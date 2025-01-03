import pandas as pd
import numpy as np
from prophet import Prophet
from pathlib import Path
from sklearn.metrics import mean_squared_error
import logging

logging.getLogger('prophet').setLevel(logging.ERROR)

def load_data(sample_size=None):
    """Load and prepare data for Prophet"""
    data_dir = Path('~/data/ml-zoomcamp-2024').expanduser()
    
    # Load datasets
    sales = pd.read_csv(data_dir / 'sales.csv', sep=';')
    online = pd.read_csv(data_dir / 'online.csv', sep=';')
    catalog = pd.read_csv(data_dir / 'catalog.csv', sep=';')
    stores = pd.read_csv(data_dir / 'stores.csv', sep=';')
    test = pd.read_csv(data_dir / 'test.csv', sep=';')
    
    # Combine sales data
    sales['channel'] = 'offline'
    online['channel'] = 'online'
    all_sales = pd.concat([sales, online], ignore_index=True)
    
    # Merge with catalog and stores
    all_sales = all_sales.merge(catalog[['item_id', 'dept_name', 'class_name']], 
                               on='item_id', how='left')
    all_sales = all_sales.merge(stores[['store_id', 'format', 'city']], 
                               on='store_id', how='left')
    
    # Convert date
    all_sales['date'] = pd.to_datetime(all_sales['date'])
    test['date'] = pd.to_datetime(test['date'])
    
    if sample_size:
        unique_combinations = all_sales[['item_id', 'store_id']].drop_duplicates()
        sampled_combinations = unique_combinations.sample(n=min(sample_size, len(unique_combinations)))
        all_sales = all_sales.merge(sampled_combinations, on=['item_id', 'store_id'])
    
    return all_sales, test

def prepare_prophet_data(df, group_cols=['item_id', 'store_id']):
    """Prepare data for Prophet model"""
    # Aggregate sales by date and group columns
    prophet_data = df.groupby(['date'] + group_cols)['quantity'].sum().reset_index()
    
    # Rename columns for Prophet
    prophet_data = prophet_data.rename(columns={'date': 'ds', 'quantity': 'y'})
    
    return prophet_data

def train_prophet_model(train_data, params=None):
    """Train Prophet model with optional parameters"""
    if params is None:
        params = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'multiplicative'
        }
    
    model = Prophet(**params)
    model.fit(train_data)
    return model

def make_predictions(model, future_dates):
    """Make predictions using trained Prophet model"""
    forecast = model.predict(future_dates)
    return forecast['yhat']

def main(sample_size=1000):
    print("Loading data...")
    all_sales, test = load_data(sample_size=sample_size)
    
    print("\nPreparing data for Prophet...")
    predictions = []
    
    # Group by store and item
    groups = all_sales.groupby(['store_id', 'item_id'])
    total_groups = len(groups)
    
    print(f"\nTraining models for {total_groups} store-item combinations...")
    for i, ((store_id, item_id), group_data) in enumerate(groups, 1):
        if i % 100 == 0:
            print(f"Processing group {i}/{total_groups}")
            
        # Prepare data for this group
        prophet_data = prepare_prophet_data(group_data)
        
        # Train model
        model = train_prophet_model(prophet_data)
        
        # Prepare future dates for this store-item combination
        future_dates = test[
            (test['store_id'] == store_id) & 
            (test['item_id'] == item_id)
        ][['date']].rename(columns={'date': 'ds'})
        
        # Make predictions
        if not future_dates.empty:
            preds = make_predictions(model, future_dates)
            
            # Store predictions
            pred_df = pd.DataFrame({
                'row_id': test[
                    (test['store_id'] == store_id) & 
                    (test['item_id'] == item_id)
                ]['row_id'],
                'quantity': preds
            })
            predictions.append(pred_df)
    
    # Combine all predictions
    final_predictions = pd.concat(predictions)
    final_predictions = final_predictions.sort_values('row_id')
    
    # Save predictions
    output_path = Path('~/data/ml-zoomcamp-2024/prophet_submission.csv').expanduser()
    final_predictions.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

if __name__ == '__main__':
    main()
