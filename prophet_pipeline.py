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
    
    # Convert date with proper format
    all_sales['date'] = pd.to_datetime(all_sales['date'])
    test['date'] = pd.to_datetime(test['date'], format='%d.%m.%Y')
    
    if sample_size:
        # Get first N unique store-item combinations
        unique_combinations = all_sales[['store_id', 'item_id']].drop_duplicates()
        first_n_combinations = unique_combinations.head(sample_size)
        print(f"Selected first {len(first_n_combinations)} unique store-item combinations")
        
        # Filter sales for these combinations
        all_sales = all_sales.merge(first_n_combinations, on=['store_id', 'item_id'])
        
        # Also filter test data
        test = test.merge(first_n_combinations, on=['store_id', 'item_id'])
    
    return all_sales, test

def prepare_prophet_data(df, group_cols=['item_id', 'store_id']):
    """Prepare data for Prophet model with proper preprocessing"""
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Ensure date is datetime
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    
    # Aggregate sales by date and group columns
    prophet_data = df.groupby(['date'] + group_cols)['quantity'].sum().reset_index()
    
    # Sort by date
    prophet_data = prophet_data.sort_values('date')
    
    # Handle negative or zero values in quantity
    prophet_data.loc[:, 'quantity'] = prophet_data['quantity'].clip(lower=0.01)
    
    # Rename columns for Prophet
    prophet_data = prophet_data.rename(columns={'date': 'ds', 'quantity': 'y'})
    
    # Add additional features that might help improve RMSE
    prophet_data.loc[:, 'month'] = prophet_data['ds'].dt.month
    prophet_data.loc[:, 'day_of_week'] = prophet_data['ds'].dt.dayofweek
    
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

def main(sample_size=None):  # Use full dataset for final predictions
    print("Loading data...")
    predictions = []
    
    try:
        all_sales, test = load_data(sample_size=sample_size)
        
        # Get unique store-item combinations from test set
        test_combinations = test[['store_id', 'item_id']].drop_duplicates()
        print(f"Found {len(test_combinations)} store-item combinations in test set")
        
        if sample_size:
            test_combinations = test_combinations.head(sample_size)
            print(f"Limiting to first {sample_size} combinations")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return []
    
    try:
        print("\nPreparing data for Prophet...")
        total_combinations = len(test_combinations)
        print(f"\nTraining models for {total_combinations} store-item combinations...")
        
        for i, (_, row) in enumerate(test_combinations.iterrows(), 1):
            store_id, item_id = row['store_id'], row['item_id']
            
            # Get training data for this combination
            group_data = all_sales[
                (all_sales['store_id'] == store_id) & 
                (all_sales['item_id'] == item_id)
            ]
                
            try:
                print(f"\nProcessing group {i}/{total_combinations}: store {store_id}, item {item_id}")
                
                # Calculate monthly averages for fallback
                monthly_avg = group_data.groupby(group_data['date'].dt.month)['quantity'].mean()
                
                if len(group_data) < 30:  # Use monthly average for insufficient data
                    print(f"Using monthly average for group {i}: insufficient data for Prophet")
                    
                    # Get test dates for this store-item combination
                    test_group = test[
                        (test['store_id'] == store_id) & 
                        (test['item_id'] == item_id)
                    ]
                    
                    if not test_group.empty:
                        # Use monthly average for predictions
                        test_months = test_group['date'].dt.month
                        preds = test_months.map(monthly_avg.to_dict()).fillna(monthly_avg.mean())
                        
                        # Store predictions
                        pred_df = pd.DataFrame({
                            'row_id': test_group['row_id'],
                            'quantity': preds
                        })
                        predictions.append(pred_df)
                else:
                    # Use Prophet for sufficient data
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
                        test_rows = test[
                            (test['store_id'] == store_id) & 
                            (test['item_id'] == item_id)
                        ]
                        pred_df = pd.DataFrame({
                            'row_id': test_rows['row_id'],
                            'quantity': preds
                        })
                        predictions.append(pred_df)
                        print(f"Made {len(pred_df)} predictions for store {store_id}, item {item_id}")
                    
            except Exception as e:
                print(f"Error processing group {i}: {str(e)}")
                continue
    
        return predictions
    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        return []

if __name__ == '__main__':
    import sys
    try:
        sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else None
        print(f"\nRunning pipeline with sample_size={sample_size}")
        predictions = main(sample_size=sample_size)
        
        if predictions and len(predictions) > 0:
            # Combine all predictions
            final_predictions = pd.concat(predictions)
            final_predictions = final_predictions.sort_values('row_id')
            
            # Verify predictions
            print(f"\nGenerated {len(final_predictions)} predictions")
            print(f"Number of unique store-item combinations: {len(predictions)}")
            print(f"Any missing values: {final_predictions['quantity'].isna().any()}")
            
            # Save predictions
            output_path = Path('~/data/ml-zoomcamp-2024/prophet_submission.csv').expanduser()
            final_predictions.to_csv(output_path, index=False)
            print(f"\nPredictions saved to {output_path}")
        else:
            print("\nNo predictions generated!")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError in main: {str(e)}")
    finally:
        print("\nPipeline execution completed")
