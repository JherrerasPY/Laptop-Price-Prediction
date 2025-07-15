# Laptop Price Prediction

This repository contains the implementation and analysis of predictive models to estimate **laptop retail prices** based on various product features such as CPU, RAM, storage type, GPU, and more. This project was developed for **COMP5310 - Principles of Data Science** at the University of Sydney.

## Project Overview

- **Research Question**:  
  What are the key attributes that influence the price of laptops in online retail markets, and how can we build a model to predict these prices?

- **Goal**:  
  Build and evaluate machine learning models to predict the retail price of laptops based on their specifications and features.

---

## Dataset

- **Rows**: 6,492 laptops  
- **Columns**: 26 attributes  
- **Attributes**: Includes numerical (e.g., CPU speed, RAM size), categorical (e.g., Manufacturer, Storage Type), and boolean features.

### Key Data Challenges:
- High proportion of **missing values** (e.g., Country of Manufacture 97% NaN)
- **Class imbalance** (e.g., 93% of laptops have Intel CPUs)
- Numerous **categorical variables** requiring encoding

---

## Preprocessing

- Dropped high-null or irrelevant columns
- Categorical encoding using:
  - `.astype("category")` for XGBoost
  - One-hot encoding for Random Forest
- Feature scaling with `StandardScaler` for numeric attributes
- Feature engineering: Created interaction terms such as `CPU_RAM_Interaction`

---

## Models Implemented

### 1. **XGBoost Regressor**
- Suitable for handling both categorical and numeric variables
- Handles overfitting well and is robust to noisy data
- **Performance**:  
  - RMSE (Test): 203.1  
  - R² (Test): 0.75  

### 2. **Random Forest Regressor**
- Provides better interpretability through feature importance
- More robust for small to medium datasets
- **Performance**:  
  - RMSE (Test): 136.5  
  - R² (Test): 0.81  

---

## Hyperparameter Tuning

- **XGBoost**: Tuned using `RandomizedSearchCV`  
  - Best: `n_estimators=300`, `max_depth=3`, `learning_rate=0.2`

- **Random Forest**: Tuned using `GridSearchCV`  
  - Best: `n_estimators=300`, `max_depth=None`, `min_samples_leaf=1`, `max_features='log2'`

---

## Evaluation Metrics

- **Root Mean Squared Error (RMSE)**  
- **R-squared (R²)**
- **Mean Absolute Error (MAE)** for Random Forest

Residual plots and prediction scatterplots were also analyzed to evaluate overfitting and prediction accuracy.

---

## 🔍 Key Findings

- **Graphics Card** was the most important feature for both models.
- Random Forest slightly outperformed XGBoost in terms of accuracy and generalization.
- Both models struggled with accurately predicting **high-priced laptops**, due to data imbalance and sparse representation.

---

## 📌 Conclusion

While both models are viable, **Random Forest** showed a better trade-off between performance and interpretability. This analysis provides actionable insights for:
- **Retailers**: Setting competitive pricing strategies
- **Consumers**: Understanding which features drive the cost of laptops

---

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [IBM: What is Random Forest?](https://www.ibm.com/topics/random-forest)
- Wade, C. (2020). *Hands-On Gradient Boosting with XGBoost and scikit-learn*
- Zhang, W. (2023). *Applied Soft Computing*
- [StackExchange Discussion on XGBoost Loss Function](https://stats.stackexchange.com/questions/202858)

