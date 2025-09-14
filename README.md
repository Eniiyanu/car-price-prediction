# DSN Bootcamp Qualification Hackathon

## Competition Overview

This repository contains my submission for the DSN Bootcamp Qualification Hackathon focused on car price prediction. The competition challenges participants to develop accurate machine learning models to predict vehicle prices based on various features and specifications.

## Evaluation Metric

Submissions are evaluated using the **Root Mean Squared Error (RMSE)** metric, which measures the difference between predicted values and actual values. The RMSE is calculated as follows:

\[
RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
\]

Where:
- \(N\) is the number of observations
- \(y_i\) is the actual value for each instance
- \(\hat{y}_i\) is the predicted value for each instance

Lower RMSE values indicate better model performance.

## Submission Format

The submission file must contain predictions for each ID in the test set with the following format:

```
id,price
188533,43878.016
188534,43878.016
188535,43878.016
```

The file should include a header row with column names "id" and "price".

## Competition Timeline

- **Start Date**: August 28, 2025
- **Entry Deadline**: September 14, 2025 (11:59 PM GMT+1)
- **Final Submission Deadline**: September 14, 2025 (11:59 PM GMT+1)

All deadlines are strict and final unless otherwise notified by the competition organizers.

## Solution Approach

My solution implements an advanced stacking ensemble model that combines multiple machine learning algorithms with proper cross-validation techniques to prevent data leakage. The approach includes:

- Comprehensive feature engineering extracting information from vehicle specifications
- Smooth target encoding for categorical variables
- A ensemble of LightGBM, XGBoost, CatBoost, Random Forest, and Extra Trees models
- Meta-model blending using Ridge regression
- Proper validation strategies to ensure generalization

## File Structure

This repository contains the following files in the root directory:

- `train.csv` - Training data with features and target prices
- `test.csv` - Test data for making predictions
- `sample_submission.csv` - Example of the required submission format
- `improved_pipeline.py` - Main Python script implementing the solution
- `README.md` - This documentation file

## Requirements

To run this code, you'll need Python 3.7+ with the following libraries:
- pandas
- numpy
- scikit-learn
- LightGBM
- XGBoost
- CatBoost

Install the required packages using:
```
pip install pandas numpy scikit-learn lightgbm xgboost catboost
```

## Usage

1. Ensure all CSV files (train.csv, test.csv, sample_submission.csv) are in the same directory as the Python script
2. Run the script with:
   ```
   python improved_pipeline.py
   ```
3. The script will generate a submission.csv file with predictions

## Notes

- This implementation focuses on both accuracy and proper validation techniques
- The model uses extensive feature engineering and a stacking ensemble approach for optimal performance
