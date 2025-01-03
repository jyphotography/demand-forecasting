import pandas as pd
import numpy as np
from pathlib import Path

def load_and_sample_data(file_path, sample_size=1000):
    """Load a sample of data from CSV file"""
    return pd.read_csv(file_path, nrows=sample_size)

def print_dataset_info(df, name):
    """Print basic information about the dataset"""
    print(f"\n=== {name} Dataset Info ===")
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\n" + "="*50)

def main():
    data_dir = Path("~/data/ml-zoomcamp-2024").expanduser()
    
    # List of files to analyze
    files = ['sales.csv', 'online.csv', 'catalog.csv', 'stores.csv', 'test.csv']
    
    for file in files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"\nAnalyzing {file}...")
            df = load_and_sample_data(file_path)
            print_dataset_info(df, file)
        else:
            print(f"File {file} not found!")

if __name__ == "__main__":
    main()
