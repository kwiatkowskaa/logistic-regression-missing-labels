# Logistic Regression with Missing Labels

## Project Overview

This project focuses on the implementation and analysis of logistic regression models in a situation where the training dataset contains observations with missing labels.

The project is divided into three main parts:

- **Task 1** – algorithms generating missing labels  
- **Task 2** – implementation of Logistic Lasso regression using the FISTA algorithm  
- **Task 3** – extension to learning with missing labels using custom strategies  


## Task 1 - TO DO


## Task 2 — Logistic Regression with FISTA

### Description

In this task, we implement Logistic Regression with L1 regularization (Logistic Lasso) using the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

The implementation:

- works on datasets **without missing values**
- uses:
  - training data: `X_train`, `y_train`
  - validation data: `X_valid`, `y_valid`
- selects the optimal λ based on validation performance  

### Supported Evaluation Metrics

The following metrics can be used for validation:

- `precision`  
- `recall`  
- `f1`  
- `balanced_accuracy`  
- `roc_auc`  
- `pr_auc`  

For threshold-based metrics, a default threshold of **0.5** is used.

### How to Run

Below is a complete example showing how to train the FISTA-based logistic regression model, perform validation-based lambda selection, and visualize the results.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- prepare data ---
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# --- initialize model ---
model = FISTALogisticLasso(
    lambda_values=np.logspace(-4, 2, 40),
    max_iter=500,
    tol=1e-6
)

# --- train with validation ---
model.fit(
    X_train,
    y_train,
    auto_validate=True,
    X_valid=X_valid,
    y_valid=y_valid,
    measure="roc_auc"
)

# --- predictions ---
proba = model.predict_proba(X_valid)
pred = model.predict(X_valid)

# --- visualization ---
model.plot()
model.plot_coefficients()

```
## Task 3 — TO DO