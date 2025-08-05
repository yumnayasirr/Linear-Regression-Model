# Smart Car Price Prediction – Linear Regression

This project implements a Linear Regression model to predict car prices based on various features. The process includes preprocessing, feature engineering, model evaluation, and interpretation.

---

## Dataset

- Source: Kaggle – Car Price Prediction
- Target: `Price`
- Features:
  - Make
  - Model
  - Year
  - Transmission
  - Fuel Type
  - Mileage
  - Engine Size

---

## Libraries Used

```python
pandas  
numpy  
matplotlib.pyplot  
seaborn  
sklearn.model_selection  
sklearn.linear_model  
sklearn.metrics  
sklearn.preprocessing  
```

---

## Preprocessing

- Label Encoding was used initially, but produced illogical coefficients.
- One-Hot Encoding was implemented using `pd.get_dummies()` which significantly improved:
  - Interpretability
  - Accuracy

---

## Model Results

| Method                | R² Score | MSE       |
|----------------------|----------|-----------|
| Label Encoded Model  | ~0.80    | 4,847,812 |
| One-Hot Encoded Model| 0.818    | 4,824,426 |
| Final Optimized Model| 0.843    | ~4.6M     |

---

## Feature Importance

Key predictors:
- `Year` (positive correlation)
- `Engine Size` (positive correlation)
- `Mileage` (negative correlation)

Features like `Make` and `Fuel Type`, despite appearing important, were dropped due to negative impact on performance.

---

## Cross-Validation

Used 5-Fold Cross Validation to ensure generalization:

```
R² Scores: [0.8204, 0.8063, 0.8332, 0.8413, 0.8554]
Average R²: 0.8313
```

---

## Polynomial Regression (Discarded)

Polynomial features were tested but resulted in lower accuracy. Linear relationships performed better.

---

## Final Model

- Model: Linear Regression (Sklearn)
- Final Features: `Year`, `Engine Size`, `Mileage`, `Model`, `Transmission`
- R² Score: 0.843

---

## Visualization

Visualizations include:
- Coefficient bar charts
- Actual vs predicted price plots

---

## Conclusion

- One-Hot Encoding led to a more logical and accurate model.
- Selecting features based on both domain logic and performance improved accuracy.
- The final model is interpretable and generalizes well.

---

## Run It

```bash
python car_pred.py
```

---

## Author

Yumna  
Computer Science Undergraduate | Machine Learning Enthusiast
