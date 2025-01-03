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
    
    # Load datasets with appropriate separators
    sales = pd.read_csv(data_dir / 'sales.csv', index_col=0)  # comma-separated
    online = pd.read_csv(data_dir / 'online.csv', index_col=0)  # comma-separated
    catalog = pd.read_csv(data_dir / 'catalog.csv', index_col=0)  # comma-separated
    stores = pd.read_csv(data_dir / 'stores.csv', index_col=0)  # comma-separated
    test = pd.read_csv(data_dir / 'test.csv', sep=';')  # semicolon-separated
    
    print(f"Loaded data shapes: sales={sales.shape}, online={online.shape}, catalog={catalog.shape}, stores={stores.shape}")
    
    # Validate required columns
    required_cols = {
        'sales': ['date', 'item_id', 'quantity', 'store_id'],
        'online': ['date', 'item_id', 'quantity', 'store_id'],
        'catalog': ['item_id', 'dept_name', 'class_name'],
        'stores': ['store_id', 'format', 'city']
    }
    
    for df_name, cols in required_cols.items():
        df = locals()[df_name]
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {df_name}: {missing_cols}")
    
    # Combine sales data
    sales['channel'] = 'offline'
    online['channel'] = 'online'
    all_sales = pd.concat([sales, online], ignore_index=True)
    
    print(f"Combined sales shape: {all_sales.shape}")
    
    # Merge with catalog and stores
    all_sales = all_sales.merge(catalog[required_cols['catalog']], 
                               on='item_id', how='left')
    all_sales = all_sales.merge(stores[required_cols['stores']], 
                               on='store_id', how='left')
    
    print(f"Final merged sales shape: {all_sales.shape}")
    
    # Convert date
    all_sales['date'] = pd.to_datetime(all_sales['date'])
    test['date'] = pd.to_datetime(test['date'])
    
    if sample_size:
        unique_combinations = all_sales[['item_id', 'store_id']].drop_duplicates()
        sampled_combinations = unique_combinations.sample(n=min(sample_size, len(unique_combinations)))
        all_sales = all_sales.merge(sampled_combinations, on=['item_id', 'store_id'])
    
    return all_sales, test

def prepare_prophet_data(df, group_cols=['item_id', 'store_id']):
    """Prepare data for Prophet model with proper preprocessing"""
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Aggregate sales by date and group columns
    prophet_data = df.groupby(['date'] + group_cols)['quantity'].sum().reset_index()
    
    # Sort by date
    prophet_data = prophet_data.sort_values('date')
    
    # Handle negative or zero values in quantity
    prophet_data['quantity'] = prophet_data['quantity'].clip(lower=0.01)
    
    # Rename columns for Prophet
    prophet_data = prophet_data.rename(columns={'date': 'ds', 'quantity': 'y'})
    
    # Add additional features that might help improve RMSE
    prophet_data['month'] = prophet_data['ds'].dt.month
    prophet_data['day_of_week'] = prophet_data['ds'].dt.dayofweek
    
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

def main(sample_size=10):  # Start with very small sample for testing
    print("Loading data...")
    predictions = []
    validation_rmse = []
    
    try:
        all_sales, test = load_data(sample_size=sample_size)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return [], []
    
    try:
        print("\nPreparing data for Prophet...")
        
        # Group by store and item
        groups = all_sales.groupby(['store_id', 'item_id'])
        total_groups = len(groups)
        
        print(f"\nTraining models for {total_groups} store-item combinations...")
        
        # Calculate validation period
        max_date = all_sales['date'].max()
        validation_start = max_date - pd.Timedelta(days=30)
        
        for i, ((store_id, item_id), group_data) in enumerate(groups, 1):
            try:
                print(f"\nProcessing group {i}/{total_groups}: store {store_id}, item {item_id}")
                
                # Split into train and validation
                train_data = group_data[group_data['date'] <= validation_start]
                val_data = group_data[group_data['date'] > validation_start]
                
                if len(train_data) < 10:  # Skip if too little data
                    print(f"Skipping group {i}: insufficient data")
                    continue
                
                # Prepare data for this group
                prophet_data = prepare_prophet_data(train_data)
                
                # Train model
                model = train_prophet_model(prophet_data)
                
                # Validate model
                if not val_data.empty:
                    val_dates = pd.DataFrame({'ds': val_data['date'].unique()})
                    val_preds = make_predictions(model, val_dates)
                    
                    # Aggregate actual values for validation
                    val_actuals = prepare_prophet_data(val_data)
                    
                    # Calculate RMSE for this group
                    group_rmse = np.sqrt(mean_squared_error(
                        val_actuals['y'],
                        val_preds[:len(val_actuals)]
                    ))
                    validation_rmse.append(group_rmse)
                    print(f"Group RMSE: {group_rmse:.4f}")
                
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
                    
            except Exception as e:
                print(f"Error processing group {i}: {str(e)}")
                continue
    
        return predictions, validation_rmse
    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        return [], []

if __name__ == '__main__':
    try:
        predictions, validation_rmse = main()  # Start with default small sample size
        
        if predictions and len(predictions) > 0:
            # Combine all predictions
            final_predictions = pd.concat(predictions)
            final_predictions = final_predictions.sort_values('row_id')
            
            # Save predictions
            output_path = Path('~/data/ml-zoomcamp-2024/prophet_submission.csv').expanduser()
            final_predictions.to_csv(output_path, index=False)
            print(f"\nPredictions saved to {output_path}")
        
        if validation_rmse and len(validation_rmse) > 0:
            mean_rmse = np.mean(validation_rmse)
            print(f"\nMean Validation RMSE: {mean_rmse:.4f}")
            print(f"Number of successful models: {len(validation_rmse)}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError in main: {str(e)}")
    finally:
        print("\nPipeline execution completed")
