"""
@author: FatemehChajaei
"""

from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict
import pandas as pd
import time
import os

from my_functions import preprocess_data, perform_grid_search
from my_functions import evaluate_performance

# Set paths
param_path = "Train_Test/RoI"

# Load training and testing data for estimating daily average temperature
df_train = pd.read_csv(os.path.join(param_path,
                                    "Train_Parameters_DataFrame.csv"))
df_test = pd.read_csv(os.path.join(param_path,
                                   "Test_Parameters_DataFrame.csv"))

# Load training and testing data for estimating daily maximum temperature
df_train_max = pd.read_csv(os.path.join(param_path,
                                        "Parameters_DataFrame_max.csv"))
df_test_max = pd.read_csv(os.path.join(param_path,
                                       "Test_Parameters_DataFrame_max.csv"))

# Load training and testing data for estimating daily minimum temperature
df_train_min = pd.read_csv(os.path.join(param_path,
                                        "Parameters_DataFrame_min.csv"))
df_test_min = pd.read_csv(os.path.join(param_path,
                                       "Test_Parameters_DataFrame_min.csv"))

# Preprocess the data
X_train, y_train, X_test, y_test = preprocess_data(df_train, df_test)
X_train_max, y_train_max, X_test_max, y_test_max = preprocess_data(
    df_train_max, df_test_max
    )
X_train_min, y_train_min, X_test_min, y_test_min = preprocess_data(
    df_train_min, df_test_min
)

# Define XGBoost model
model = XGBRegressor()

# Define hyperparameter grid for Grid Search
param_grid = dict(n_estimators=[100, 250, 500, 750, 1000],
                  max_depth=[3, 5, 7, 10],
                  eta=[0.001, 0.01, 0.1],
                  min_child_weight=[1, 3, 5, 10],
                  subsample=[0.5, 0.75, 1],
                  colsample_bytree=[0.5, 0.75, 1])

# Perform Grid Search
best_params_avg = perform_grid_search(model, param_grid,
                                      X_train, y_train)
best_params_max = perform_grid_search(model, param_grid,
                                      X_train_max, y_train_max)
best_params_min = perform_grid_search(model, param_grid,
                                      X_train_min, y_train_min)

model_avg = XGBRegressor(**best_params_avg)
model_max = XGBRegressor(**best_params_max)
model_min = XGBRegressor(**best_params_min)

# Use KFold cross-validation
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# Train and predict on the training set
start = time.time()
df_train["y_pred"] = cross_val_predict(model_avg, X_train,
                                       y_train, cv=cv)
print("Time elapsed for training (Average Temp): ", time.time() - start)

start = time.time()
df_train_max["y_pred"] = cross_val_predict(model_max, X_train_max,
                                           y_train_max, cv=cv)
print("Time elapsed for training (Max Temp): ", time.time() - start)

start = time.time()
df_train_min["y_pred"] = cross_val_predict(model_min, X_train_min,
                                           y_train_min, cv=cv)
print("Time elapsed for training (Min Temp): ", time.time() - start)

# Train and predict on the test set
start = time.time()
df_test["y_pred"] = cross_val_predict(model_avg, X_test, y_test, cv=cv)
print("Time elapsed for testing (Average Temp): ", time.time() - start)

start = time.time()
df_test_max["y_pred"] = cross_val_predict(model_max, X_test_max,
                                          y_test_max, cv=cv)
print("Time elapsed for testing (Max Temp): ", time.time() - start)

start = time.time()
df_test_min["y_pred"] = cross_val_predict(model_min, X_test_min,
                                          y_test_min, cv=cv)
print("Time elapsed for testing (Min Temp): ", time.time() - start)

# Evaluate performance
evaluate_performance(df_train, "Train (Average Temp)")
evaluate_performance(df_train_max, "Train (Max Temp)")
evaluate_performance(df_train_min, "Train (Min Temp)")

evaluate_performance(df_test, "Test (Average Temp)")
evaluate_performance(df_test_max, "Test (Max Temp)")
evaluate_performance(df_test_min, "Test (Min Temp)")
