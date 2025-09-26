This repository contains scripts and experiments in Machine Learning, created for educational purposes. It includes practical examples of data preprocessing (such as working
with categorical features, handling missing values in both numerical and categorical variables, etc.), building pipelines, regression and classification models, predictions
and model evaluations using metrics like MAE. The purpose of this repository is to document my learning process in ML, experimenting with different models and methodologies.
The main libraries that I work with are scikit-learn, pandas, XGBoost, matplotlib and seaborn, among others.

--> Repository structure:

    -MLPipeline.py:   simple piece of code that analyzes both training and test datasets, automatically detects missing values, categorical features and the target feature,
                      to then create a Pipeline that uses a SimpleImputer ('mean' strategy for numerical, 'most_frequent' for categorical missing values) and OneHotEncoding
                      with categorical features. Once preprocessed, data is fed to an XGBoost model with already set parameters to create predictions.

--> Libraries used:

    -Scikit-learn
    
    -XGBoost
    
    -Matplotlib
