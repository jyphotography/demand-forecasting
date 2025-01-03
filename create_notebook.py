import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create cells
cells = [
    nbf.v4.new_markdown_cell("# Retail Demand Forecasting Analysis\n\nThis notebook contains the analysis for the ML Zoomcamp 2024 Competition focusing on retail demand forecasting."),
    
    nbf.v4.new_markdown_cell("## Data Preparation\n\nIn this section, we will:\n1. Load and examine the raw data\n2. Check for missing values\n3. Analyze data types and basic statistics\n4. Handle any data format issues"),
    
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option("display.max_columns", None)
plt.style.use("seaborn")"""),
    
    nbf.v4.new_markdown_cell("## Data Cleaning\n\nIn this section, we will:\n1. Remove or handle outliers\n2. Handle missing values\n3. Format dates and categorical variables\n4. Create any necessary derived features"),
    
    nbf.v4.new_code_cell("# Data loading and initial examination will go here\n# Code will be added once we have access to the dataset")
]

nb.cells = cells

# Write the notebook to a file
with open("notebooks/notebook.ipynb", "w") as f:
    nbf.write(nb, f)
