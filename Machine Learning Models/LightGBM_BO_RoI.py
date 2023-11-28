# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 14:36:24 2023

@author: NovinGostar
"""

from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import pandas as pd
import numpy as np
import time
import os


param_path = r"D:\Code\Amsterdam Parameter\Train_Test\RoI"
save_path = r"D:\Code\Temp_Prediction_Amsterdam\RoI\Models\save_models"

df = pd.read_csv(os.path.join(param_path, "Train_Parameters_DataFrame.csv"))
df_test = pd.read_csv(os.path.join(param_path, "Test_Parameters_DataFrame.csv"))

X = df[df.columns[1:15]]
X_test = df_test[df_test.columns[1:15]]


df_all = np.concatenate((X, X_test))

scaler = MinMaxScaler()
Xs = scaler.fit_transform(df_all)

y = df.loc[:,['Temp']]
y = y.to_numpy()
y = np.ravel(y)
Xtrain = Xs[0:8060]

y_test = df_test.loc[:,['Temp']]
y_test = y_test.to_numpy()
y_test = np.ravel(y_test)
Xtest = Xs[8060:]


cv = KFold(n_splits=10, random_state=1, shuffle=True)

param_hyperopt= {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 2500, 50)),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 50, 1)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 1, 50, 1)),
    'reg_lambda': hp.uniform('reg_lambda', 0.01, 1.0),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'subsample': hp.uniform('subsample', 0.01, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.01, 1.0), 
    # 'reg_lambda': hp.uniform('log-uniform', 1e-9, 1),      # L2 regularization
    # 'reg_alpha': hp.uniform('log-uniform', 1e-9, 1),       # L1 regularization
    }



def hyperopt(param_space, X_train, y_train, num_eval):
    
    start = time.time()
    
    def objective_function(params):
        reg = LGBMRegressor(**params)
        score = cross_val_score(reg, Xtrain, y, cv=5).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function, 
                      param_space, 
                      algo=tpe.suggest, 
                      max_evals=num_eval, 
                      trials=trials,
                      rstate= np.random.default_rng(1))
    loss = [x['result']['loss'] for x in trials.trials]
    
    #best_param_values = [x for x in best_param.values()]
    
    if best_param['boosting_type'] == 0:
        boosting_type = 'gbdt'
    else:
        boosting_type= 'dart'
    
 
    
    reg_best = LGBMRegressor(learning_rate=best_param['learning_rate'],
                                  max_depth=int(best_param['max_depth']),
                                  n_estimators=int(best_param['n_estimators']),
                                  num_leaves=int(best_param['num_leaves']),
                                  boosting_type=boosting_type,
                                  colsample_bytree=best_param['colsample_bytree'],
                                  subsample=best_param['subsample'],
                                  reg_lambda=best_param['reg_lambda'])
    reg_best.fit(Xtrain, y)
    
    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss)*-1)
    print("Best parameters: ", best_param)
    print("Time elapsed: ", time.time() - start)
    print("Parameter combinations evaluated: ", num_eval)
    
    return trials, reg_best

results_hyperopt, reg_best = hyperopt(param_hyperopt, Xtrain, y, 20)


reg_best = LGBMRegressor(learning_rate=0.045,
                              max_depth=21,
                              n_estimators=3000,
                              num_leaves=28,
                              boosting_type='gbdt',
                              colsample_bytree=0.921596100038763,
                              subsample=0.21744055366203258,
                              reg_lambda=0.7329500731542772)

reg_best.fit(Xtrain, y)



df_test["y_pred"] = reg_best.predict(Xtest)
df["y_pred"] = reg_best.predict(Xtrain)

RMSE_train = np.round(mean_squared_error(df["Temp"], df["y_pred"], squared=False),3)
MAE_train = np.round(mean_absolute_error(df["Temp"], df["y_pred"]),3)
RMSE_test = np.round(mean_squared_error(df_test["Temp"], df_test["y_pred"], squared=False),3)
MAE_test = np.round(mean_absolute_error(df_test["Temp"], df_test["y_pred"]),3)
print ("RMSE Train: ", RMSE_train, "      RMSE Test: ", RMSE_test)
print ("MAE Train:  ", MAE_train, "      MAE Test: ", MAE_test)



from sklearn.metrics import r2_score
r2 = r2_score(df_test["Temp"], df_test["y_pred"])
print('r2 score for perfect model is', r2)

# df_test_out = df_test[['PointID', 'X', 'Y', 'DoY', 'Temp' ,'y_pred']].copy()
# df_test_out.to_csv(os.path.join(r"D:\UNI\M\Thesis\2\SFW_Code\Amsterdam Parameter\5x5 Param\Final", 'LightGBM_100m_TestPoints.csv'),index=False)

# df_test_july18 = df_test_out[df_test_out.DoY == 199]
# df_test_july18.to_csv(os.path.join(r"D:\UNI\M\Thesis\2\SFW_Code\Amsterdam Parameter\5x5 Param\Final", 'LightGBM_100m_TestPoints_July18.csv'),index=False)

df_test.to_csv(os.path.join(r"D:\UNI\M\Thesis\2\SFW_Code\Amsterdam Parameter\5x5 Param\Final", 'LightGBM_100m_TestPoints.csv'),index=False)




