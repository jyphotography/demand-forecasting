import nbformat as nbf

def create_cell_content():
    cells = []
    
    # Add data loading section
    cells.append(nbf.v4.new_markdown_cell("## Data Loading\n\nLoading all CSV files from the competition dataset:"))
    
    # Add code to load all CSV files
    load_code = '''# Load all CSV files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn')

# Load all CSV files
actual_matrix = pd.read_csv('data/actual_matrix.csv')
catalog = pd.read_csv('data/catalog.csv')
discounts = pd.read_csv('data/discounts_history.csv')
markdowns = pd.read_csv('data/markdowns.csv')
online = pd.read_csv('data/online.csv')
price_history = pd.read_csv('data/price_history.csv')
sales = pd.read_csv('data/sales.csv')
stores = pd.read_csv('data/stores.csv')
test = pd.read_csv('data/test.csv')

# Display basic information about each dataset
for name, df in {
    'Actual Matrix': actual_matrix,
    'Catalog': catalog,
    'Discounts History': discounts,
    'Markdowns': markdowns,
    'Online': online,
    'Price History': price_history,
    'Sales': sales,
    'Stores': stores,
    'Test': test
}.items():
    print(f"\\n{name} Dataset:")
    print(f"Shape: {df.shape}")
    print("\\nColumns:")
    print(df.columns.tolist())
    print("\\nSample data:")
    print(df.head(2))
    print("\\nData Info:")
    df.info()
    print("\\n" + "="*50)'''
    
    cells.append(nbf.v4.new_code_cell(load_code))
    
    # Add initial analysis section
    cells.append(nbf.v4.new_markdown_cell("""## Initial Data Analysis

Key aspects to analyze:
1. Data completeness (missing values)
2. Data types and potential type conversions
3. Value distributions
4. Temporal patterns (for time-series data)
5. Relationships between different datasets"""))
    
    # Add code for missing values analysis
    missing_values_code = '''# Analyze missing values in each dataset
print("Missing Values Analysis:\\n")

for name, df in {
    'Actual Matrix': actual_matrix,
    'Catalog': catalog,
    'Discounts History': discounts,
    'Markdowns': markdowns,
    'Online': online,
    'Price History': price_history,
    'Sales': sales,
    'Stores': stores,
    'Test': test
}.items():
    missing = df.isnull().sum()
    if missing.any():
        print(f"\\n{name} Dataset Missing Values:")
        print(missing[missing > 0])
    else:
        print(f"\\n{name} Dataset: No missing values")'''
    
    cells.append(nbf.v4.new_code_cell(missing_values_code))
    
    # Add basic visualization section
    cells.append(nbf.v4.new_markdown_cell("## Data Visualization\n\nLet's create some basic visualizations to understand our data better:"))
    
    viz_code = '''# Set up the plotting style
plt.style.use('seaborn')
sns.set_palette("husl")

# 1. Store Distribution Analysis
plt.figure(figsize=(10, 6))
stores['store_type'].value_counts().plot(kind='bar')
plt.title('Distribution of Store Types')
plt.xlabel('Store Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Sales Overview
plt.figure(figsize=(12, 6))
sales.groupby('date')['sales'].sum().plot(kind='line')
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=price_history, x='price', bins=50)
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 4. Online vs Offline Analysis
if 'online' in online.columns:  # Check if 'online' column exists
    plt.figure(figsize=(8, 6))
    online['online'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Online vs Offline Distribution')
    plt.axis('equal')
    plt.show()'''
    
    cells.append(nbf.v4.new_code_cell(viz_code))
    
    return cells

# Read existing notebook
with open('notebooks/notebook.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# Add new cells
nb.cells.extend(create_cell_content())

# Write updated notebook
with open('notebooks/notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
