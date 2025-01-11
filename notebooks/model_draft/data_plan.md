# Data Manipulation Plan for Sales Prediction

## 1. Data Preprocessing Steps

### 1.1 Fix Data Format Issues
- Parse test.csv properly (currently semicolon-separated)
- Convert date columns to datetime format in all relevant files
- Handle missing values in catalog.csv (particularly weight_volume, weight_netto, fatness)

### 1.2 Feature Engineering
- Create time-based features:
  - Day of week
  - Month
  - Is weekend
  - Is holiday (if possible to obtain holiday data)
  - Day of month
  - Season
- Create lag features from historical sales:
  - Previous day sales
  - Previous week sales
  - Previous month sales
  - Rolling averages (7-day, 30-day)
- Create store-based features:
  - Store size category (based on area)
  - Store format encoding
  - City encoding
- Create product-based features:
  - Department encoding
  - Class encoding
  - Subclass encoding
  - Price category (binned price_base)
  - Average item sale quantity
  - Item popularity score

### 1.3 Data Integration
- Merge sales.csv and online.csv to get total sales per item/store/date
- Join with catalog.csv to add product characteristics
- Join with stores.csv to add store information
- Create aggregated features at different levels:
  - Store level
  - Department level
  - Class level
  - Time level

## 2. Data Cleaning
- Remove outliers based on quantity and price (use IQR method)
- Handle missing values:
  - For categorical: create "Unknown" category
  - For numerical: impute with median or mean based on similar items
- Normalize/scale numerical features
- Encode categorical variables:
  - Use label encoding for ordinal categories
  - Use one-hot encoding for nominal categories

## 3. Data Validation
- Check for data leakage
- Verify date continuity
- Ensure all required features are available for test set
- Validate engineered features
- Check correlations between features
- Verify scaling/normalization results

## 4. Train/Validation Split Strategy
- Use time-based split
- Last month of training data as validation set
- Ensure validation period matches test set characteristics

## 5. Feature Selection
- Use correlation analysis
- Apply feature importance from tree-based models
- Remove redundant features
- Focus on features available in test set

## 6. Data Pipeline Implementation
1. Load raw data
2. Clean and preprocess
3. Engineer features
4. Handle missing values
5. Scale/normalize
6. Encode categorical variables
7. Create final feature matrix

## 7. Monitoring and Validation
- Track data quality metrics
- Monitor feature distributions
- Validate assumptions about data relationships
- Check for data drift between train and test sets
