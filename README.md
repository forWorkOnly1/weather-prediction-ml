

# Rainfall Prediction for Melbourne

A machine learning project that builds a classifier to predict daily rainfall in Melbourne, Australia using historical weather data.

**Final Model Accuracy: 84%**

## Project Overview

This project demonstrates a complete ML pipeline to solve a binary classification problem: "Will it rain today in Melbourne?" It covers data cleaning, feature engineering, model training with hyperparameter tuning, and comprehensive evaluation.

## Dataset & Preprocessing

- **Source:** Australian Bureau of Meteorology data (2008-2017)
- **Key Preprocessing Steps:**
  - Focused on Melbourne area only
  - Engineered `Season` feature from dates
  - Handled class imbalance (76% "No Rain" vs 24% "Rain")
  - Used stratified sampling for fair evaluation

## Methodology

- **Algorithm:** Random Forest Classifier
- **Pipeline:** Automated preprocessing (scaling + one-hot encoding) with `ColumnTransformer`
- **Validation:** 5-fold Stratified Cross-Validation
- **Hyperparameter Tuning:** Grid Search for optimal `n_estimators`, `max_depth`, and `min_samples_split`

## Results

**Best Model Performance:**
- Cross-validation Score: 86%
- Test Set Accuracy: 84%

**Detailed Classification:**
- **No Rain Prediction:** 95% recall (excellent at identifying dry days)
- **Rain Prediction:** 51% recall (moderate at detecting actual rain)
- **Overall:** Strong performance with room for improvement on rain detection

## Technologies Used

- Python, pandas, scikit-learn
- Random Forest, GridSearchCV
- matplotlib, seaborn for visualization

## Quick Start

```bash
# Install dependencies
pip install pandas scikit-learn matplotlib seaborn

# Run the Jupyter notebook
jupyter notebook FinalProject_AUSWeather.ipynb
```

## Key Insights

- The model significantly outperforms the naive baseline (76% accuracy)
- Seasonality proved to be one of the most important features
- Class imbalance is the main challenge for improving rain detection

The project provides a solid foundation for weather prediction and demonstrates practical machine learning application on real-world data.
