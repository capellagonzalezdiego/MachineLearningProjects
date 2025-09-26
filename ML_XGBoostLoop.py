# This code builds a loop to find the best parameters for an XGBoost model.
 
 
# It starts off by asking the user to introduce both databases and saving them. This first version is only be able to read .csv archives, but it can be
# improved to accept many others.


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
# dataset. In future versions, we will ask the user to choose with which features wants to work with.


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


# In this case, we are working with some datasets that don't have any missing values, nor categorical features. Let's create the loop.


import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

y = train_data[target_feature]
X = train_data.drop(columns=[target_feature[0]])

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=1)

learning_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
results = []

for lr in learning_rates:
    model_lr = XGBRegressor(n_estimators = 1000, learning_rate = lr, early_stopping_rounds = 15)
    model_lr.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds_lr = model_lr.predict(X_val)
    scores_lr = mean_absolute_error(y_val, preds_lr)
    print(f"For lr={lr:.2f} -> best_iteration={model_lr.best_iteration+1}, MAE={scores_lr:.8f}")
    results.append({"learning_rate": lr, "best_n_estimators": model_lr.best_iteration+1, "MAE": scores_lr})

# We save the results in a DataFrame to choose the learning_rates value that results in the minimun MAE:

results_df = pd.DataFrame(results)
print(results_df)
best_row = results_df.loc[results_df["MAE"].idxmin()]
best_lr = best_row["learning_rate"]
best_n = int(best_row["best_n_estimators"])
best_mae = best_row["MAE"]
print(f"The best learning rate is {best_lr}, which results in a model with {best_n} estimators and a MAE of {best_mae}.")

# Let's visualize the results:

import matplotlib.pyplot as plt

plt.plot(results_df['learning_rate'], results_df['MAE'])
plt.scatter(best_lr, best_mae, color='red', s=100, label=f'Best: {best_lr:.3f}')
plt.xlabel('Learning Rate of the XGBoost model')
plt.ylabel('MAE')
plt.title('MAE vs. Learning Rate')
plt.savefig("MAE_vs._LearningRate - XGBoostBPM2.png")
plt.close()


# Now that we have found the best parameters for our XGBoost model, we create the final one, train it with the whole training dataset, and generate 
# the prediction for the test dataset.

final_model = XGBRegressor(n_estimators = best_n, learning_rate = best_lr, n_jobs = -1, random_state = 1)
final_model.fit(X,y)
final_pred = final_model.predict(test_data)