from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np


def preprocess_data(df_train, df_test):
    """Preprocess the data."""
    df_all = pd.concat([df_train[df_train.columns[1:15]],
                        df_test[df_test.columns[1:15]]])

    # Use MinMaxScaler for feature scaling
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(df_all)

    # Extract target variable 'Temp'
    y_train = np.ravel(df_train.loc[:, ['Temp']].to_numpy())
    X_train = Xs[:len(df_train)]

    y_test = np.ravel(df_test.loc[:, ['Temp']].to_numpy())
    X_test = Xs[len(df_train):]

    return X_train, y_train, X_test, y_test


def perform_grid_search(model, param_grid, X, y):
    """Perform Grid Search to find the best hyperparameters."""
    reg = GridSearchCV(model, param_grid, verbose=2)
    search = reg.fit(X, y)
    print('----------------------------')
    print(search.best_params_)
    print('----------------------------')
    return search.best_params_


def evaluate_performance(df_eval, dataset_name):
    """Evaluate and print the performance metrics."""
    RMSE = np.round(mean_squared_error(df_eval["Temp"], df_eval["y_pred"],
                                       squared=False), 3)
    MAE = np.round(mean_absolute_error(df_eval["Temp"], df_eval["y_pred"]), 3)

    print(f"{dataset_name} RMSE: {RMSE}      {dataset_name} MAE: {MAE}")

    MAPE = mean_absolute_percentage_error(df_eval["Temp"], df_eval["y_pred"])

    print(f"{dataset_name} MAPE: {MAPE}")

    r2 = r2_score(df_eval["Temp"], df_eval["y_pred"])
    print(f'{dataset_name} r2 score: {r2}')
