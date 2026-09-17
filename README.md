# Credit Card Fraud Detection

A supervised machine learning project that detects fraudulent credit card transactions using multiple classification models, compared via cross-validation and tuned with `GridSearchCV`.

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (originally from Kaggle / ULB Machine Learning Group). It contains 284,807 transactions made by European cardholders, of which only 492 (~0.17%) are fraudulent — making this a highly imbalanced classification problem.

Features `V1`–`V28` are the result of a PCA transformation (for confidentiality); `Time` and `Amount` are the only original, untransformed features. The target column is `Class` (`1` = fraud, `0` = legitimate).

> **Note:** The dataset (`creditcard.csv`) is not included in this repo due to its size and Kaggle's terms of use. Download it from the link above and place it in the project root before running the notebook.

## Approach

1. **Explore & inspect** the data — shape, dtypes, null check via `data.info()`.
2. **Split** into train/test sets (`test_size=0.33`) before any model fitting.
3. **Compare multiple models** with 10-fold cross-validation on accuracy:
   - Logistic Regression (`class_weight='balanced'`)
   - K-Nearest Neighbors
   - Gaussian Naive Bayes
   - Decision Tree (`class_weight='balanced'`)
4. **Evaluate with `classification_report`** (precision/recall/F1), since accuracy alone is misleading on imbalanced data — a model predicting "not fraud" every time would still score ~99.8% accuracy.
5. **Select Decision Tree** as the best-performing model based on its balance of precision and recall on the minority (fraud) class.
6. **Tune with `GridSearchCV`**, optimizing for **F1 score** (not accuracy) across `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, and `max_features`.
7. **Final evaluation** on the held-out test set with the tuned model.

## Results

| Model | CV Accuracy | Fraud-class Precision | Fraud-class Recall | Fraud-class F1 |
|---|---|---|---|---|
| Logistic Regression | 0.963 | 0.04 | 0.92 | 0.08 |
| KNN | 0.998 | 1.00 | 0.03 | 0.06 |
| Naive Bayes | 0.993 | 0.14 | 0.66 | 0.23 |
| Decision Tree (baseline) | 0.999 | 0.70 | 0.73 | 0.72 |
| **Decision Tree (tuned)** | — | **0.89** | **0.79** | **0.83** |

Best hyperparameters found: `{'criterion': 'entropy', 'max_depth': 5, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 2}`

Final tuned model: **99.95% test accuracy**, **0.83 F1-score** on the fraud class.

## Requirements

```
numpy
pandas
seaborn
scikit-learn
xgboost
```

Install with:
```bash
pip install -r requirements.txt
```

## Usage

1. Download `creditcard.csv` from Kaggle and place it in the project root.
2. Open `creditcard.ipynb` in Jupyter Notebook / JupyterLab / VS Code.
3. Run all cells in order.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
