# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 21:58:06 2019

@author: Stefan Borkovski
"""
# This code is used for training and testing Gradient Boosting Regressor model for 3 and 6 DOF 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor as gbr

# loading the data sets

###          3DOF 

# df = pd.read_csv(r'.\dataset_3DOF\dataset_with_constraints_no6_3WRIST_15K.csv',  encoding = 'utf8')
# MAE [0.10624026581956693, 0.15616805447615276, 0.13938704291074547]

###         6DOF (dataset with constraints no 6/7)

df = pd.read_csv(r'.\datasets_6DOF\dataset_myplace_with_constraints_no6_merged_plus3.csv',  encoding = 'utf8')
# MAE [3.819868259782215, 1.871342541554445, 2.08383743340577, 5.863820251341452, 3.7049899703536595, 22.548858767211478]              cross_val [0.10860406592656537, 0.11720947321502871, 0.1094927504079676, 0.12328082332514634, 0.05459377018100175, 0.16212063413281178]

df = df.drop(['Unnamed: 0'], axis = 1)

# number of features
size = 12 # end
# configure this parameter to 3 or 6 degrees of freedom
angles = 6 # 

# creating scalers which will be used for scaling the data and returning it back to the normal state
x_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
y_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))

X = df.iloc[:,:size]; 
y = df.iloc[:,size:]; 

X_s = x_scaler.fit_transform(X)
y_s = y_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2)

X_train = pd.DataFrame(X_train); X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train); y_test = pd.DataFrame(y_test)

#           Gradient Boosting Regressor start

# training the model

from sklearn.ensemble import GradientBoostingRegressor as gbr
gbr = gbr(loss='ls', learning_rate=0.1, n_estimators = 800)    

gbr.fit(X_train, y_train[0])
y_pred1 = np.reshape(gbr.predict(X_test),(X_test.shape[0],1))

gbr.fit(X_train, y_train[1])
y_pred2 = np.reshape(gbr.predict(X_test),(X_test.shape[0],1))

gbr.fit(X_train, y_train[2])
y_pred3 = np.reshape(gbr.predict(X_test),(X_test.shape[0],1))

gbr.fit(X_train, y_train[3])
y_pred4 = np.reshape(gbr.predict(X_test),(X_test.shape[0],1))

gbr.fit(X_train, y_train[4])
y_pred5 = np.reshape(gbr.predict(X_test),(X_test.shape[0],1))

gbr.fit(X_train, y_train[5])
y_pred6 = np.reshape(gbr.predict(X_test),(X_test.shape[0],1))

y_pred = np.concatenate((y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6), axis = 1)
#y_pred = np.concatenate((y_pred1, y_pred2, y_pred3), axis = 1)

# unscaling the data
y_pred = pd.DataFrame(y_scaler.inverse_transform(y_pred))
y_test = pd.DataFrame(y_scaler.inverse_transform(y_test))

#           Gradient Boosting Regressor end

result_mse = []
result_mae = []

# measuring model performance
for i in range(angles):

   mse = mean_squared_error(y_test.iloc[:,i], y_pred.iloc[:,i])
   rmse = sqrt(mse) 
   mae = mean_absolute_error(y_test.iloc[:,i], y_pred.iloc[:,i])
   result_mse.append(mse)
   result_mae.append(mae)

print("RMSE", result_mse)
print("MAE", result_mae)

X_test = pd.DataFrame(x_scaler.inverse_transform(X_test))

# saving results

# y_pred.to_csv(r'.\y_pred.csv')
# y_test.to_csv(r'.\y_test.csv')
# X_test.to_csv(r'.\X_test.csv')


