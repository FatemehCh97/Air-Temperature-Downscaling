# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 03:42:03 2023

@author: NovinGostar
"""

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
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


"""Grid search-Kfold"""
model =SVR()
distributions = dict(kernel = ['sigmoid', 'rbf', 'linear', 'poly'], 
                      C = [0.01, 0.1, 1, 100])

reg = GridSearchCV(model, distributions, verbose=2)
search = reg.fit(Xtrain, y)
print('----------------------------')
print(search.best_params_)
print('----------------------------')


model = SVR(kernel='rbf', C=100, gamma='auto', epsilon=.1)

model.fit(Xtrain, y)

df_test["y_pred"] = model.predict(Xtest)
df["y_pred"] = model.predict(Xtrain)

RMSE_train = np.round(mean_squared_error(df["Temp"], df["y_pred"], squared=False),3)
MAE_train = np.round(mean_absolute_error(df["Temp"], df["y_pred"]),3)
RMSE_test = np.round(mean_squared_error(df_test["Temp"], df_test["y_pred"], squared=False),3)
MAE_test = np.round(mean_absolute_error(df_test["Temp"], df_test["y_pred"]),3)
print ("RMSE Train: ", RMSE_train, "      RMSE Test: ", RMSE_test)
print ("MAE Train:  ", MAE_train, "      MAE Test: ", MAE_test)

APE = (np.abs(df_test["Temp"]-df_test["y_pred"]))/(df_test["Temp"])
MAPE = mean_absolute_percentage_error(df_test["Temp"], df_test["y_pred"])
print("MAPE: ", MAPE)


from sklearn.metrics import r2_score
r2 = r2_score(df_test["Temp"], df_test["y_pred"])
print('r2 score for perfect model is', r2)
