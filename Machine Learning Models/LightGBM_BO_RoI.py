"""
@author: FatemehChajaei
"""

from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import pandas as pd
import numpy as np
import time
import os

from my_functions import preprocess_data, evaluate_performance

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

cv = KFold(n_splits=10, random_state=1, shuffle=True)

param_hyperopt = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 2500, 50)),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 50, 1)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 1, 50, 1)),
    'reg_lambda': hp.uniform('reg_lambda', 0.01, 1.0),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'subsample': hp.uniform('subsample', 0.01, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.01, 1.0),
    # 'reg_lambda': hp.uniform('log-uniform', 1e-9, 1), # L2 regularization
    # 'reg_alpha': hp.uniform('log-uniform', 1e-9, 1), # L1 regularization
    }


def hyperopt(param_space, X_train, y_train, num_eval):

    start = time.time()

    def objective_function(params):
        reg = LGBMRegressor(**params)
        score = cross_val_score(reg, X_train, y_train, cv=5).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()

    best_param = fmin(objective_function,
                      param_space,
                      algo=tpe.suggest,
                      max_evals=num_eval,
                      trials=trials,
                      rstate=np.random.default_rng(1))

    loss = [x['result']['loss'] for x in trials.trials]

    # best_param_values = [x for x in best_param.values()]

    if best_param['boosting_type'] == 0:
        boosting_type = 'gbdt'
    else:
        boosting_type = 'dart'

    reg_best = LGBMRegressor(learning_rate=best_param['learning_rate'],
                             max_depth=int(best_param['max_depth']),
                             n_estimators=int(best_param['n_estimators']),
                             num_leaves=int(best_param['num_leaves']),
                             boosting_type=boosting_type,
                             colsample_bytree=best_param['colsample_bytree'],
                             subsample=best_param['subsample'],
                             reg_lambda=best_param['reg_lambda'])

    reg_best.fit(X_train, y_train)

    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss)*-1)
    print("Best parameters: ", best_param)
    print("Time elapsed: ", time.time() - start)
    print("Parameter combinations evaluated: ", num_eval)

    return trials, reg_best


results_hyperopt, reg_best = hyperopt(param_hyperopt, X_train, y_train, 20)
results_hyperopt_max, reg_best_max = hyperopt(param_hyperopt, X_train_max,
                                              y_train_max, 20)
results_hyperopt_min, reg_best_min = hyperopt(param_hyperopt, X_train_min,
                                              y_train_min, 20)

# Train LightGBM model
reg_best.fit(X_train, y_train)
reg_best_max.fit(X_train_max, y_train_max)
reg_best_min.fit(X_train_min, y_train_min)

# Train and predict on the training set
df_train["y_pred"] = reg_best.predict(X_train)
df_train_max["y_pred"] = reg_best_max.predict(X_train)
df_train_min["y_pred"] = reg_best_min.predict(X_train)

# Train and predict on the test set
df_test["y_pred"] = reg_best.predict(X_test)
df_test_max["y_pred"] = reg_best_max.predict(X_test)
df_test_min["y_pred"] = reg_best_min.predict(X_test)

# Evaluate performance
evaluate_performance(df_train, "Train (Average Temp)")
evaluate_performance(df_train_max, "Train (Max Temp)")
evaluate_performance(df_train_min, "Train (Min Temp)")

evaluate_performance(df_test, "Test (Average Temp)")
evaluate_performance(df_test_max, "Test (Max Temp)")
evaluate_performance(df_test_min, "Test (Min Temp)")
