import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_analyze_predictions():
    """Load and analyze the predictions"""
    data_dir = Path('~/data/ml-zoomcamp-2024').expanduser()
    
    # Load predictions and sample submission
    predictions = pd.read_csv(data_dir / 'submission.csv')
    test = pd.read_csv(data_dir / 'test.csv', sep=';')
    
    print("\nPredictions Analysis:")
    print("-" * 50)
    print(f"Total predictions: {len(predictions):,}")
    print(f"\nQuantity Statistics:")
    print(predictions['quantity'].describe())
    
    # Check for invalid predictions
    print("\nInvalid Predictions:")
    print(f"Negative quantities: {(predictions['quantity'] < 0).sum():,}")
    print(f"Zero quantities: {(predictions['quantity'] == 0).sum():,}")
    print(f"Very large quantities (>1000): {(predictions['quantity'] > 1000).sum():,}")
    
    # Plot distribution of predictions
    plt.figure(figsize=(10, 6))
    plt.hist(predictions['quantity'], bins=50)
    plt.title('Distribution of Predicted Quantities')
    plt.xlabel('Quantity')
    plt.ylabel('Count')
    plt.savefig(data_dir / 'prediction_distribution.png')
    plt.close()
    
    # Compare with test data structure
    print("\nTest Data Info:")
    print(f"Total test cases: {len(test):,}")
    print("\nUnique values in test data:")
    for col in test.columns:
        print(f"{col}: {test[col].nunique():,} unique values")

if __name__ == '__main__':
    load_and_analyze_predictions()
