import pandas as pd
import numpy as np

def analyze_data():
    # Load test data and split the combined column
    test = pd.read_csv('test.csv')
    # Split the combined column into separate columns
    test = pd.DataFrame([x.split(';') for x in test.iloc[:, 0]], 
                       columns=['row_id', 'item_id', 'store_id', 'date'])
    
    try:
        predictions = pd.read_csv('prophet_submission.csv')
    except Exception as e:
        print(f"\nError loading predictions: {e}")
        predictions = None
    
    print('\nTest set info:')
    print(test.info())
    
    print('\nUnique combinations in test:')
    unique_combos = test.groupby(['store_id', 'item_id']).size().reset_index()
    print(f'Shape: {unique_combos.shape}')
    print('\nFirst few combinations:')
    print(unique_combos.head())
    
    print('\nSample of test data:')
    print(test.head())
    
    if predictions is not None:
        print('\nPredictions analysis:')
        print(f'Total rows: {len(predictions)}')
        print(f'Missing values:\n{predictions.isna().sum()}')
        print(f'Unique row_ids: {predictions.row_id.nunique()}')
        
        print('\nSample predictions:')
        print(predictions.head())
        
        # Check for any mismatches between test and predictions
        test_rows = set(test['row_id'])
        pred_rows = set(predictions['row_id'])
        missing_rows = test_rows - pred_rows
        extra_rows = pred_rows - test_rows
        
        print('\nRow ID analysis:')
        print(f'Missing row_ids: {len(missing_rows)}')
        print(f'Extra row_ids: {len(extra_rows)}')
        if len(missing_rows) > 0:
            print('\nSample missing row_ids:')
            print(sorted(list(missing_rows))[:5])
    
    # Print distribution of predictions per store-item combination
    print('\nPredictions per store-item combination:')
    predictions_per_combo = test.groupby(['store_id', 'item_id']).size()
    print(predictions_per_combo.value_counts().sort_index())

if __name__ == '__main__':
    analyze_data()
