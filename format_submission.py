import pandas as pd
from pathlib import Path

def format_submission():
    """Format submission file according to competition requirements"""
    data_dir = Path('~/data/ml-zoomcamp-2024').expanduser()
    
    # Load current submission and sample submission
    submission = pd.read_csv(data_dir / 'submission.csv')
    sample_submission = pd.read_csv(data_dir / 'sample_submission.csv')
    
    # Verify columns match requirements
    print("\nVerifying submission format...")
    print(f"Required columns: {', '.join(sample_submission.columns)}")
    print(f"Current columns: {', '.join(submission.columns)}")
    
    # Ensure row_ids match
    print("\nVerifying row_ids...")
    submission_rows = set(submission['row_id'].astype(str))
    sample_rows = set(sample_submission['row_id'].astype(str))
    
    missing_rows = sample_rows - submission_rows
    extra_rows = submission_rows - sample_rows
    
    print(f"Missing row_ids: {len(missing_rows)}")
    print(f"Extra row_ids: {len(extra_rows)}")
    
    if len(missing_rows) > 0:
        print("First few missing row_ids:", list(missing_rows)[:5])
    if len(extra_rows) > 0:
        print("First few extra row_ids:", list(extra_rows)[:5])
    
    # Format final submission
    final_submission = submission[['row_id', 'quantity']]
    
    # Save formatted submission
    output_path = data_dir / 'final_submission.csv'
    final_submission.to_csv(output_path, index=False)
    print(f"\nFormatted submission saved to {output_path}")
    
    # Display sample of final submission
    print("\nSample of final submission:")
    print(final_submission.head())

if __name__ == '__main__':
    format_submission()
