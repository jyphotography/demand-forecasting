# Retail Demand Forecasting

This project is part of the [ML Zoomcamp 2024 Competition](https://www.kaggle.com/competitions/ml-zoomcamp-2024-competition/overview), focusing on predicting customer demand based on historical sales data from multiple retail stores.

## Problem Description

The challenge involves forecasting customer demand for products across multiple stores using 25 months of historical sales data. The model will help optimize stock management and reduce operational inefficiencies in retail operations.

## Environment Setup

### Prerequisites
- Python 3.12 or higher
- pip (Python package installer)

### Setting up the Virtual Environment

1. Clone the repository:
```bash
git clone https://github.com/jyphotography/demand-forecasting.git
cd demand-forecasting
```

2. Create and activate the virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
## On Windows:
venv\Scripts\activate
## On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Jupyter Notebook
After activating the virtual environment, start Jupyter:
```bash
jupyter notebook
```

## Project Structure
- `notebooks/`: Jupyter notebooks for EDA and model development
- `src/`: Source code for the final model
- `requirements.txt`: Project dependencies
