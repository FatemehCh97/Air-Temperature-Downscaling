"""
@author: FatemehChajaei
"""

from sklearn.svm import SVR
import pandas as pd
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

# Define Support Vector Regression (SVR) model
model = SVR()

# Define hyperparameter grid for Grid Search
param_grid = dict(kernel=['sigmoid', 'rbf', 'linear', 'poly'],
                  C=[0.01, 0.1, 1, 100])

# Perform Grid Search
best_params_avg = perform_grid_search(model, param_grid,
                                      X_train, y_train)
best_params_max = perform_grid_search(model, param_grid,
                                      X_train_max, y_train_max)
best_params_min = perform_grid_search(model, param_grid,
                                      X_train_min, y_train_min)

model_avg = SVR(**best_params_avg)
model_max = SVR(**best_params_max)
model_min = SVR(**best_params_min)

# Train SVR model
model_avg.fit(X_train, y_train)
model_max.fit(X_train_max, y_train_max)
model_min.fit(X_train_min, y_train_min)

# Train and predict on the training set
df_train["y_pred"] = model_avg.predict(X_train)
df_train_max["y_pred"] = model_max.predict(X_train)
df_train_min["y_pred"] = model_min.predict(X_train)

# Train and predict on the test set
df_test["y_pred"] = model_avg.predict(X_test)
df_test_max["y_pred"] = model_max.predict(X_test)
df_test_min["y_pred"] = model_min.predict(X_test)

# Evaluate performance
evaluate_performance(df_train, "Train (Average Temp)")
evaluate_performance(df_train_max, "Train (Max Temp)")
evaluate_performance(df_train_min, "Train (Min Temp)")

evaluate_performance(df_test, "Test (Average Temp)")
evaluate_performance(df_test_max, "Test (Max Temp)")
evaluate_performance(df_test_min, "Test (Min Temp)")
