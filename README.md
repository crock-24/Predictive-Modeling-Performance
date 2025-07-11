# Nonlinear Regression Model Comparison in R

## Overview
This project compares the performance of several nonlinear and linear regression models using the `caret` package in R. The models are trained and evaluated on a chemical manufacturing dataset, as well as a simulated dataset from `mlbench.friedman1()`. The goal is to identify the most accurate model and understand the influence of hyperparameters and feature importance.

---

## Models Implemented

### Nonlinear Models
- **Radial Basis Function Support Vector Machine (SVM)**
- **Multivariate Adaptive Regression Splines (MARS)**
- **K-Nearest Neighbors (KNN)**
- **Partial Least Squares Regression (PLS)**

### Linear Models
- **Multiple Linear Regression (LM)**
- **Lasso Regression**

---

## Key Results

| Model                      | Tuning Parameters              | RMSE     |
|---------------------------|--------------------------------|----------|
| Radial SVM                | Cost = 8                       | **1.1041** |
| MARS                      | Degree = 2, Terms = 8          | 1.1637   |
| Lasso (Linear)            | Lambda = 0.0621                | **1.1449** |
| Partial Least Squares     | Components = 3                 | 1.2224   |
| K Nearest Neighbors       | Neighbors = 11                 | 1.4634   |
| Multiple Linear Regression| N/A                            | 3.2825   |

---

## 💡 Insights
- **SVM** had the best performance among nonlinear models.
- **Lasso** was the top-performing linear model, very close to SVM.
- **MARS** performed competitively and correctly identified the informative predictors (X1–X5).
- Predictor importance analysis shows **process variables dominate** over biological ones.

---

1. required packages:
```r
install.packages(c(
  "caret", "kernlab", "earth", "mlbench", 
  "pls", "lars", "AppliedPredictiveModeling", "VIM"
))
