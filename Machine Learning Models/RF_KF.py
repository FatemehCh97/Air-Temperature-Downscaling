# -*- coding: utf-8 -*-
"""
@author: FatemehChajaei
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import os


param_path = r"D:\Code\Amsterdam Parameter\Train_Test\RoI"
save_path = r"D:\Code\Temp_Prediction_Amsterdam\RoI\Models\save_models"

df = pd.read_csv(os.path.join(param_path, "Train_Parameters_DataFrame.csv"))
df_test = pd.read_csv(os.path.join(param_path, "Test_Parameters_DataFrame.csv"))

# df = pd.read_csv(os.path.join(param_path, "Parameters_DataFrame_2.csv"))
# df_test = pd.read_csv(os.path.join(param_path, "Test_Parameters_DataFrame_2.csv"))


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

model = RandomForestRegressor()

"""Grid search-Kfold"""

distributions = dict(n_estimators = [100, 500, 1000], 
                      max_depth=[3, 5, 7, 8, 10],
                      max_features=[0.5, 0.7, 0.8, 1])

reg = GridSearchCV(model, distributions, verbose=2)
search = reg.fit(Xtrain, y)
print('----------------------------')
print(search.best_params_)
print('----------------------------')


model = RandomForestRegressor(n_estimators= search.best_params_['n_estimators'],
                      max_depth= search.best_params_['max_depth'],
                      max_features= search.best_params_['max_features'])

model = RandomForestRegressor(n_estimators = 500,
                      max_depth = 5,
                      max_features = 0.7)

model.fit(Xtrain, y)

########## Train
df["y_pred"] = model.predict(Xtrain)
df_test["y_pred"] = model.predict(Xtest)


RMSE_train = np.round(mean_squared_error(df["Temp"], df["y_pred"], squared=False),3)
MAE_train = np.round(mean_absolute_error(df["Temp"], df["y_pred"]),3)
RMSE_test = np.round(mean_squared_error(df_test["Temp"], df_test["y_pred"], squared=False),3)
MAE_test = np.round(mean_absolute_error(df_test["Temp"], df_test["y_pred"]),3)
print ("RMSE Train: ", RMSE_train, "      RMSE Test: ", RMSE_test)
print ("MAE Train:  ", MAE_train, "      MAE Test:  ", MAE_test)


APE = (np.abs(df_test["Temp"]-df_test["y_pred"]))/(df_test["Temp"])
MAPE = mean_absolute_percentage_error(df_test["Temp"], df_test["y_pred"])
print("MAPE: ", MAPE)


from sklearn.metrics import r2_score
r2 = r2_score(df_test["Temp"], df_test["y_pred"])
print('r2 score for perfect model is', r2)
