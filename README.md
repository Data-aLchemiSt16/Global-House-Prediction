# Global-House-Prediction
This project uses the Global House Purchase Decision Dataset from Kaggle, which contains various factors influencing house buying decisions.  Source: https://www.kaggle.com/datasets/mohankrishnathalla/global-house-purchase-decision-dataset

# Predicting Customer House Purchases: A Case Study in Detecting and Correcting Data Leakage

# Project Overview

This project focuses on building a binary classification model to predict whether a customer will purchase a house. The primary business goal is to help a sales team identify and prioritize high-potential leads, thereby optimizing their efforts and increasing conversion rates.

The core narrative of this project is not just about building a model, but about the critical data science process of identifying and correcting **data leakage**. An initial model achieved a misleading 100% accuracy, leading to an investigation that revealed flawed features. The final result is a realistic, honest model that provides actionable business insights.

## Data Leakage Visual :
!Feature Importance PLot for analyzing problem(

# The Data Leakage Story: From a "Perfect" Model to a Realistic One

A key challenge and learning experience in this project was dealing with data leakage, where the model had access to information that would not be available at the time of prediction.

1. The "Too Good to Be True" Model
My initial Random Forest model achieved a perfect **100% accuracy**, with flawless precision and recall. While seemingly a success, this is a major red flag in any real-world machine learning task.

**Initial Flawed Feature Importances:**
*The model's predictions were almost entirely based on features that were only available *after* a purchase decision was made.*

![Leaky Feature Importance Plot]('C:\Users\AVI SHARMA\Documents\work\insights\output.png')
*(This plot clearly shows `satisfaction_score` and `emi_to_income_ratio` as top predictors—a logical impossibility for a predictive model.)*

### 2. The Investigation and Correction
The investigation revealed that features like `satisfaction_score`, `loan_amount`, and `emi_to_income_ratio` were **post-purchase information**. A customer can't have a satisfaction score for a purchase they haven't made yet.

The solution was to remove these leaky features and rebuild the entire preprocessing and modeling pipeline on a feature set that would realistically be available to the sales team.

### 3. The Realistic and Actionable Model
After removing the leaky features, the new model provides a much more honest assessment of our predictive power.

| Metric             | Flawed Model (with Leakage) | **Realistic Model (Corrected)** |
| ------------------ | --------------------------- | ------------------------------- |
| **Accuracy**       | 100%                        | **74%**                         |
| **AUC Score**      | 1.00                        | **0.77**                        |
| **Recall (Buy=1)** | 100%                        | **17%**                         |

This realistic model, while having lower overall accuracy, is infinitely more valuable as it is built on a sound methodology and provides genuine insights.

---

## Key Visualizations from the Realistic Model

#### Final Feature Importances
*Financial health (`disposable_income`, `customer_salary`) and property characteristics (`legal_cases`, `crime_cases`) are now the key drivers of the decision.*

![Realistic Feature Importance Plot](images/realistic_feature_importance.png](https://github.com/Data-aLchemiSt16/Global-House-Prediction/blob/main/Feature%20importance.png)

#### Model Performance
*The model is strong at identifying non-buyers but struggles to find potential buyers (low recall), highlighting a clear area for future improvement.*

![Realistic Confusion Matrix](images/realistic_confusion_matrix.png](https://github.com/Data-aLchemiSt16/Global-House-Prediction/blob/main/Confusion%20Matrix.png)

---

## Technologies Used
- **Python 3.9**
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, Imbalanced-Learn (for SMOTE)
- **Data Visualization:** Matplotlib, Seaborn
- **Development Environment:** Jupyter Notebook
