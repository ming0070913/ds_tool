# Introduction
This is a tool kit for data science project, using demo data from Kaggle. This tool kit can automatically analyze the column type in a dataset, perform cleansing and one-hot encoding accordingly. It also performs training and evalaution of basic machine learning models, such as decision tree and random forest; and produce ROC curve and visialization on the prediction performance.

This is useful for quick pre-processing and analysis of a dataset. It's useful for real world business project, when there are a large number of columns in a dataset. The visualization also helps to spot columns with data leakage problem. This tool was inspired when doing real world business project and using some expensive commerical data processing tool. 

# Structure
`Visualize.ipynb`: Visualizing a dataset

`Analyze.ipynb`: Train and evaluate ML models on dataset

`cleanser.py`: Automated data pre-processing and cleansing functions

`analyzer.py`: ML models evaluating functions

`requirements.txt`: Python package dependency and versions

`README.md`: This file

# Dependency
Python packages that can be installed with pip
 - matplotlib==3.0.3
 - numpy==1.16.3
 - pandas==0.24.2
 - pydotplus==2.0.2
 - scikit-learn==0.21.1
 - seaborn==0.9.0

Packages needed to be install in the system for Windows:
 - graphviz==0.10.1

# Data set
The demo data used is from Kaggle **_Telecom Customer Churn Prediction_** dataset. (Size: 46.2 MB, Link: https://www.kaggle.com/abhinav89/telecom-customer/downloads/telecom-customer.zip/1)
