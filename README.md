# MammoMind: Cutting-Edge ML for Early Tumor Detection

## Overview

This project demonstrates the development and evaluation of machine learning models for classifying breast cancer tumors as benign or malignant using the Breast Cancer Wisconsin dataset (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data). The primary goal is to build a predictive model with high accuracy, ROC-AUC, and clinical relevance that can assist in early cancer detection.

Key components of the project include:

* Data loading and exploration
* Data preprocessing and feature engineering
* Feature selection (variance threshold, univariate selection, and RFE)
* Model training, evaluation, and comparison (Random Forest, Gradient Boosting, AdaBoost, Logistic Regression, SVM, KNN, Decision Tree, and Neural Network)
* Visualization of performance metrics and feature importance
* Identification of the best-performing model and detailed analysis

## Dataset

* **Source**: Breast Cancer Wisconsin dataset (569 samples, 30 original features) from UCI Machine Learning Repository.
* **Features**: Measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension for the tumor’s mean, standard error, and worst values.
* **Target**: `diagnosis` column with labels: `M` (malignant) and `B` (benign).
* **Class Distribution**: 357 benign (62.7%) and 212 malignant (37.3%) samples.

## Installation and Requirements

1. **Clone the repository** or download project files.
2. **Environment**:

   * Python 3.7+
   * Jupyter Notebook or Google Colab (optional but recommended)
3. **Dependencies** (install via pip):

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

## File Structure

```
├── data.csv                # Breast Cancer Wisconsin dataset
├── 422_research.pdf         # A concise, up‑to‑date literature review of leading research (2023 – 2025) on breast‑cancer prediction using machine learning, sourced with the help of Perplexity AI.
├── CSE422_[AI]_projectl.ipynb # Jupyter notebook with code implementation
└── README.md               # This file
```

> **Note**: Update paths in code (if needed) to correctly point to `data.csv` before running the notebook.

## Usage

1. **Open the Jupyter Notebook**:

   * Launch `CSE422_[AI]_project.ipynb` in Jupyter or upload it to Google Colab.
2. **Run all cells** sequentially:

   * Step 1: Load and explore the dataset
   * Step 2: Data preprocessing and feature engineering
   * Step 3: Feature analysis and selection
   * Step 4: Data splitting and scaling
   * Step 5: Model training and evaluation
   * Step 6: Results comparison and visualization
   * Step 7: Best model analysis
3. **Review outputs**:

   * Tabulated performance metrics (accuracy, ROC-AUC, cross-validation)
   * Plots: correlation heatmap, model performance bar charts, feature importance, confusion matrix, and ROC curve
   * Classification report of the best model (Neural Network - MLP)

## Methodology

### 1. Data Loading and Exploration

* Load `data.csv` into a pandas DataFrame.
* Inspect shape (569 rows, 33 columns), column names, and missing values.
* Identify that the dataset contains an unnecessary `id` column and an empty `Unnamed: 32` column.
* Examine target distribution to ensure class imbalance is manageable.

### 2. Data Preprocessing and Feature Engineering

* **Drop columns**: Remove `id` and `Unnamed: 32`.
* **Label Encoding**: Convert `diagnosis` (M/B) to numerical labels (1/0) using `LabelEncoder`.
* **Separate Features and Target**:

  * `X`: 30 features (all numeric measurements).
  * `y`: Encoded diagnosis labels.

### 3. Feature Analysis and Selection

* **Descriptive Statistics**: Summarize feature scales and distributions via `DataFrame.describe()`.
* **Correlation Matrix**: Visualize pairwise correlations using a heatmap to identify redundancy.
* **Variance Threshold**: Remove features with low variance (threshold = 0.1) to drop near-constant features (reduces from 30 to 11 features).
* **Univariate Selection**: Apply `SelectKBest` with ANOVA F-test to select top 20 features.
* **Recursive Feature Elimination (RFE)**:

  * Use a `RandomForestClassifier` as the estimator.
  * Select top 15 features based on recursive elimination:

    * `['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 'area_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst']`.
* **Final Feature Set**: Use RFE-selected 15 features for downstream modeling to maximize predictive power while reducing dimensionality.

### 4. Data Splitting and Scaling

* **Train-Test Split**: Stratified split (80% train, 20% test) to preserve class distribution (455 training samples, 114 testing samples).
* **Scaling**: Apply `StandardScaler` to training features and transform test features for algorithms sensitive to feature scales (e.g., logistic regression, SVM, KNN, MLP).

### 5. Model Training and Evaluation

Train multiple classifiers on the processed data:

* **Tree-based Models (unscaled features)**:

  * Random Forest
  * Gradient Boosting
  * AdaBoost
  * Decision Tree
* **Linear/Distance-based Models (scaled features)**:

  * Logistic Regression
  * Support Vector Classifier
  * K-Nearest Neighbors
* **Neural Network**:

  * Multi-Layer Perceptron (MLP) with hidden layers (100, 50)

#### Metrics Computed per Model:

* **Accuracy** on test set
* **ROC-AUC** on test set
* **5-Fold Cross-Validation** mean and standard deviation on the training set

**Summary of Model Performance** (sorted by accuracy):

| Model                     | Accuracy | ROC-AUC | CV Mean | CV Std |
| ------------------------- | -------- | ------- | ------- | ------ |
| Neural Network (MLP)      | 0.9912   | 0.9987  | 0.9670  | 0.0197 |
| Support Vector Classifier | 0.9825   | 0.9970  | 0.9692  | 0.0128 |
| K-Nearest Neighbors       | 0.9825   | 0.9980  | 0.9560  | 0.0209 |
| Random Forest             | 0.9737   | 0.9952  | 0.9560  | 0.0311 |
| Logistic Regression       | 0.9737   | 0.9980  | 0.9736  | 0.0112 |
| Gradient Boosting         | 0.9649   | 0.9950  | 0.9538  | 0.0306 |
| AdaBoost                  | 0.9649   | 0.9914  | 0.9670  | 0.0269 |
| Decision Tree             | 0.9035   | 0.8938  | 0.9297  | 0.0420 |

### 6. Results Visualization

* **Accuracy and ROC-AUC Bar Charts**: Compare model performance side-by-side.
* **Cross-Validation Scores**: Bar chart with error bars showing variability.
* **Feature Importance**: Plot top 10 important features from the best performing tree-based model (Random Forest), highlighting key predictors.
* **Confusion Matrix** and **ROC Curve**: For the best model (MLP), visualize classification performance and AUC.

### 7. Best Model Analysis

* The **Neural Network (MLP)** achieved the highest test accuracy (99.12%) and ROC-AUC (0.9987).
* **Classification Report** for MLP on test set (114 samples):

  * Benign (72 samples): precision 0.99, recall 1.00, f1-score 0.99
  * Malignant (42 samples): precision 1.00, recall 0.98, f1-score 0.99
* **Confusion Matrix**:

  * True negatives (benign correctly identified): 72
  * True positives (malignant correctly identified): 41
  * False negatives (missed malignant): 1
  * False positives (benign incorrectly labeled malignant): 0
* **ROC Curve**: AUC = 0.9987, demonstrating extremely high discriminative ability.

## Key Insights

* Ensemble methods (Random Forest, Gradient Boosting, AdaBoost) and the Neural Network model perform exceptionally well (≥96% accuracy).
* Feature selection (RFE) effectively reduces dimensionality to 15 features without sacrificing predictive performance.
* The MLP achieves a clinical-grade accuracy (99.12%) with a low false negative rate, crucial for minimizing missed cancer diagnoses.

## Clinical Implications

* A robust predictive model can support radiologists by providing a second opinion on mammography data.
* Early and accurate detection of malignant tumors increases the chances of successful treatment and improved patient outcomes.
* While the results are promising, clinical deployment requires regulatory approval and validation on diverse external datasets.

## Limitations and Important Notes

* **Data Source**: The model is trained and tested solely on the Wisconsin dataset; real-world performance may vary on different populations and imaging techniques.
* **Regulatory Compliance**: For use in a healthcare setting, compliance with FDA, CE, and other regional regulations is mandatory.
* **Explainability**: Black-box models like neural networks lack transparency; future work should focus on explainable AI to improve trust among clinicians.
* **Ethical Considerations**: Privacy, data security, and informed consent must be addressed before clinical integration.

## Next Steps

1. **External Validation**: Test the model on independent datasets (e.g., other hospitals, multi-center studies) to assess generalizability.
2. **Explainable AI**: Integrate techniques such as SHAP or LIME to interpret model predictions and highlight important features at the sample level.
3. **Clinical Integration**:

   * Develop a user-friendly interface for radiologists to upload mammography features and receive predictions.
   * Collaborate with healthcare providers to pilot the model in a clinical workflow.
4. **Prospective Studies**: Conduct prospective clinical trials to evaluate the model’s real-world performance and impact on patient outcomes.

## How to Cite This Project

If you use this work in academic research or clinical evaluation, please cite:

> A. K. M. N. Kabir, "Breast Cancer Classification Project," GitHub repository, 2025. [Online]. Available: https://github.com/nihal-kabir/CSE422-Artificial-Intelligence-Project



---

*Last updated: May 31, 2025*
