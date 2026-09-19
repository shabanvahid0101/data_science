# House Price Prediction — Ames Housing Dataset

Predicting residential home sale prices using a Random Forest Regressor, built as part of my applied data science learning path.

## Overview

This project uses the [Ames Housing dataset](https://www.kaggle.com/c/home-data-for-ml-course) (Kaggle's "Home Data for ML Course" competition) to predict the sale price of houses in Ames, Iowa, based on a set of numerical property features.

## Dataset

- **Source:** Kaggle — Home Data for ML Course
- **Target variable:** `SalePrice`
- **Features used:**
  - `LotArea` — Lot size in square feet
  - `YearBuilt` — Original construction year
  - `1stFlrSF` — First floor square footage
  - `2ndFlrSF` — Second floor square footage
  - `FullBath` — Number of full bathrooms
  - `BedroomAbvGr` — Bedrooms above grade
  - `TotRmsAbvGrd` — Total rooms above grade (excludes bathrooms)

## Approach

1. **Data loading & feature selection** — loaded the training data with `pandas` and selected 7 numerical features as predictors (`X`), with `SalePrice` as the target (`y`).
2. **Train/validation split** — split the data using `train_test_split` to evaluate the model honestly on unseen data.
3. **Model training** — trained a `RandomForestRegressor` from scikit-learn, which combines many decision trees to produce more stable, accurate predictions than a single tree.
4. **Evaluation** — measured performance using Mean Absolute Error (MAE) on the validation set.
5. **Final model** — retrained the Random Forest on the *full* training dataset (not just the split) to maximize the data available before generating final predictions.
6. **Prediction & submission** — applied the trained model to the competition's test set and generated a `submission.csv` file in the required format.

## Results

| Model | Validation MAE |
|---|---|
| Random Forest Regressor | **$21,857** |

On average, the model's price predictions were off by about $21,857 from the true sale price — a solid baseline result for a first model using only 7 basic numerical features.

## Tech Stack

- Python
- pandas
- scikit-learn (`RandomForestRegressor`, `train_test_split`, `mean_absolute_error`)

## How to Run

1. Download the dataset from the [Kaggle competition page](https://www.kaggle.com/c/home-data-for-ml-course).
2. Place `train.csv` and `test.csv` in an `input/` folder alongside the notebook.
3. Open `house-price-prediction.ipynb` in Jupyter or Kaggle Notebooks and run all cells.
4. The final predictions are saved to `submission.csv`.

## What I Learned

- How to train and evaluate a regression model with scikit-learn
- The difference between validating on a held-out split vs. training on full data before final predictions
- Why ensemble methods (Random Forest) tend to generalize better than a single Decision Tree
- How underfitting and overfitting trade off, and how to reason about model complexity

## Next Steps

- Add more features (categorical variables, missing-value handling) to improve accuracy
- Compare Random Forest against other models (Gradient Boosting, Linear Regression)
- Tune hyperparameters (`n_estimators`, `max_depth`) with cross-validation

---
*Part of my self-directed data science learning path — built while completing Kaggle's [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) course.*
