# This code builds a Machine Learning pipeline with an XGBoost model to generate predictions from training and test datasets. It automatically
# detects the target feature, and preprocesses categorical features with OneHot Encoding, and handles missing values with SimpleImputer, using 
# averages for numerical values and the most frequent ones in categorical values. Then, data is fed to an XGBoostRegressor, which is evaluated
# with cross-validation, using Mean Absolute Error (MAE) as metric. Finally, predictions are generated for the test dataset.


# Firstly, it asks the user to introduce both training and test databases. Then, it automatically finds the target feature, and checks if there are
# any missing values or categorical features.


import pandas as pd
import os

print(f"Current working directory: ", os.getcwd())
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

train_file = input("Enter the training dataset file name (For example: train.csv): ")
test_file = input("Enter the training dataset file name (For example: test.csv): ")

missing_files = []  # Here, we check if any files are missing.
if not os.path.exists(train_file):
    missing_files.append(train_file)
if not os.path.exists(test_file):
    missing_files.append(test_file)

if missing_files:
    print("The following files were not found: ")
    for f in missing_files:
        print(f"  -{f}")
    print("Check the files and try again.")
    print(f"Current working directory: ", os.getcwd())
    print("Make sure the .csv files are in the working directory.")
else:
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    print("Datasets loaded successfully.")
    print(f"Training dataset: {train_data.shape[0]} rows and {train_data.shape[1]} columns.")
    print(f"Test dataset: {test_data.shape[0]} rows and {test_data.shape[1]} columns.")  # This line of code right here shows the dimensions of both databases.


# Now that our datasets are loaded, we will preproccess the data, which means that we are looking for missing values and categorical features. For the features
# used to make predictions, we will assume that every feature is used, except the target. We will deduct that the target feature is the one missing in the test 
# dataset.


train_columns = train_data.columns.tolist()
test_columns = test_data.columns.tolist()

target_feature = [col for col in train_columns if col not in test_columns]
if len(target_feature) == 1:
    print("Target feature is: ", target_feature)
else:
    print("Target feature not found, check databases.")


numeric_feat = []
categorical_feat = []

for col in train_columns:
    if pd.api.types.is_numeric_dtype(train_data[col]):
        numeric_feat.append(col)
    else:
        categorical_feat.append(col)

if len(categorical_feat) > 0:
    print("We have the following categorical features: ", categorical_feat)
else:
    print("There are not any categorical features in the training dataset.")


missing_feat = [col for col in train_columns if train_data[col].isna().sum() > 0]

if missing_feat:
    print("The following features have missing values: ", missing_feat)
else:
    print("There are not any features with missing values in the training dataset.")


# The next step involves creating the pipeline.

target = target_feature[0]
numeric_feat = [col for col in numeric_feat if col != target]
categorical_feat = [col for col in categorical_feat if col != target]

X = train_data[numeric_feat + categorical_feat]
y = train_data[target]

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer()
categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers = [
    ('num', numerical_transformer, numeric_feat),
    ('cat', categorical_transformer, categorical_feat)
])

import xgboost
from xgboost import XGBRegressor

pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators = 300, learning_rate = 0.075, max_depth = 4, random_state = 0))
])

pipeline.fit(X,y)

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"MAE scores: {scores}")
print(f"Average MAE scores across experiments: {scores.mean()}")

final_pred = pipeline.predict(test_data)