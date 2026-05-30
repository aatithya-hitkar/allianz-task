#!/usr/bin/env python
# coding: utf-8

# # TeleConnect Churn Prediction - Part 1
# This notebook contains the complete data cleaning, exploratory data analysis (EDA), and model building pipeline for TeleConnect's customer churn prediction.

# In[2]:


import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "..", "src"))
from data_processing import run_cleaning_pipeline, build_quality_summary
from model_building import compute_feature_associations, plot_eda_charts, engineer_features, prepare_train_test, train_logistic, train_xgboost, evaluate_model, plot_confusion_matrix, plot_roc_curves, plot_feature_importance, export_model_artifacts

import warnings
warnings.filterwarnings('ignore')


# ## 1.1 Data Quality Assessment and Cleaning
# The marketing team provided a raw dataset with known quality issues (e.g., system migrations, manual entry errors).
# Our cleaning pipeline systematically addresses impossible values (like negative age), normalizes categorical strings, corrects cross-field logical conflicts, and imputes missing data. Below is the summary table of the corrections.

# In[3]:


raw_file = '../data/test_datafile.csv'
df_cleaned = run_cleaning_pipeline(raw_file)
print("\n--- Quality Summary ---")
build_quality_summary(raw_file, df_cleaned)


# ## 1.2 Exploratory Data Analysis
# To understand the relationship between our features and churn, we compute the point-biserial correlation for all numeric and encoded categorical features.
# We also visualize three key relationships:
# 1. **Contract Type**: Month-to-month contracts have the highest churn.
# 2. **Tenure**: Early-tenure customers are at the highest risk.
# 3. **Satisfaction Score**: Low satisfaction directly doubles the churn probability.

# In[4]:


associations = compute_feature_associations(df_cleaned)
plot_eda_charts(df_cleaned)


# ### Feature Engineering
# We engineer two new features to improve model performance:
# 1. **`charges_per_month_ratio`**: Normalizes total lifetime spend by the active tenure to catch bill shock.
# 2. **`risk_score_composite`**: A simple weighted blend of satisfaction, contract type, and tenure, enabling retention reps to quickly glance at a non-model risk score.

# In[5]:


df_engineered = engineer_features(df_cleaned)


# ## 1.3 Model Building and Evaluation
# We train two distinct models:
# - **Logistic Regression**: A linear model that provides excellent baseline interpretability. Class imbalance is addressed using `class_weight='balanced'`.
# - **XGBoost**: A powerful tree-based ensemble model capable of capturing complex non-linear relationships. We address class imbalance via `scale_pos_weight`.
# 
# **Metric Selection**: For churn prediction, **Recall** is arguably the most critical metric. Missing a churner (False Negative) results in lost revenue, whereas a False Positive only results in an unnecessary retention offer (a low-cost intervention).

# In[6]:


X_train, X_test, y_train, y_test, scaler = prepare_train_test(df_engineered)

lr_model = train_logistic(X_train, y_train)
xgb_model = train_xgboost(X_train, y_train)

print("Evaluating Logistic Regression:")
lr_proba = evaluate_model("Logistic Regression", lr_model, X_test, y_test)

print("\nEvaluating XGBoost:")
xgb_proba = evaluate_model("XGBoost", xgb_model, X_test, y_test)


# ## 1.4 Visualizations
# Below are the visualizations evaluating the performance of the XGBoost model and comparing the ROC curves of both models.

# In[7]:


plot_confusion_matrix(xgb_model, X_test, y_test, "XGBoost")
plot_roc_curves({"Logistic Regression": lr_proba, "XGBoost": xgb_proba}, y_test)
plot_feature_importance(xgb_model, X_train.columns.tolist())


# ## 1.5 Export Model for Part 2
# Finally, we save the trained models, the scaler, and the feature lists as `.pkl` artifacts so they can be loaded by the retention agent in Part 2.

# In[8]:


export_model_artifacts(xgb_model, lr_model, scaler, X_train.columns.tolist())


# In[ ]:




