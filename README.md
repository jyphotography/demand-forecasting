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

## Data Setup

### Downloading Competition Data

1. Install the Kaggle API package:
```bash
pip install kaggle
```

2. Set up Kaggle credentials:
   - Go to your Kaggle account settings (https://www.kaggle.com/settings)
   - Click on "Create New API Token" to download `kaggle.json`
   - Create the Kaggle config directory:
     ```bash
     mkdir -p ~/.kaggle
     ```
   - Move the downloaded `kaggle.json` to the config directory:
     ```bash
     mv path/to/kaggle.json ~/.kaggle/
     ```
   - Set appropriate permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. Download competition data:
```bash
# Create data directory
mkdir -p data

# Download competition data
kaggle competitions download -c ml-zoomcamp-2024-competition -p data/

# Unzip the downloaded files
cd data && unzip ml-zoomcamp-2024-competition.zip && cd ..
```

## Project Structure
- `data/`: Competition dataset
- `notebooks/`: Jupyter notebooks for EDA and model development
- `src/`: Source code for the final model
- `requirements.txt`: Project dependencies
