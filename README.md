# Breast Cancer Classification — Wisconsin Diagnostic Dataset

Comparative evaluation of eight supervised ML classifiers for binary tumour classification (Benign/Malignant) using the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

## Results

| Model | Test Accuracy | ROC-AUC | CV Mean | CV Std |
|---|---|---|---|---|
| **Neural Network (MLP)** | **99.12%** | **0.9987** | 0.9670 | 0.0197 |
| Support Vector Classifier | 98.25% | 0.9970 | 0.9692 | 0.0128 |
| K-Nearest Neighbours | 98.25% | 0.9980 | 0.9560 | 0.0209 |
| Random Forest | 97.37% | 0.9952 | 0.9560 | 0.0311 |
| Logistic Regression | 97.37% | 0.9980 | 0.9736 | 0.0112 |
| Gradient Boosting | 96.49% | 0.9950 | 0.9538 | 0.0306 |
| AdaBoost | 96.49% | 0.9914 | 0.9670 | 0.0269 |
| Decision Tree | 90.35% | 0.8938 | 0.9297 | 0.0420 |

## Pipeline

- **Preprocessing:** dropped non-informative columns (`id`, `Unnamed: 32`), label-encoded target
- **Feature selection:** RFE with a 100-tree Random Forest, reduced 30 → 15 features
- **Split:** stratified 80/20 (455 train / 114 test)
- **Scaling:** `StandardScaler` applied to SVC, KNN, Logistic Regression, and MLP; tree-based models trained on unscaled data
- **Evaluation:** test accuracy, ROC-AUC, 5-fold cross-validation

## Tech Stack

- Python, scikit-learn, pandas, NumPy, Matplotlib

