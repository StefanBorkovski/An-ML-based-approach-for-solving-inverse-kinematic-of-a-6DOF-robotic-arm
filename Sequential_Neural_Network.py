# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:27:53 2019

@author: Stefan Borkovski
"""

# This code is used for training and testing Sequential Neural Network model for 3 and 6 DOF

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import *
from keras.callbacks import EarlyStopping

# loading the data sets

###          3DOF 

# df = pd.read_csv(r'.\dataset_3DOF\dataset_with_constraints_no6_3WRIST_15K.csv',  encoding = 'utf8')
#   MAE [0.10624026581956693, 0.15616805447615276, 0.13938704291074547]
#   cross_val MAE [0.16358137192021718, 0.1487296730002861, 0.15814576777707862]


###          6DOF

df = pd.read_csv(r'.\datasets_6DOF\dataset_myplace_with_constraints_no6_merged_plus3.csv',  encoding = 'utf8')
#   MAE [1.5180150620045452, 1.0686760142594887, 1.4876093738307, 1.636541801932513, 1.4278333895332755, 22.798384573224745]
#   cross_val[1.5129320383351554, 0.9476663252798746, 1.0964469020573984, 1.5888469189977055, 1.0088474341799742, 21.822538931056535]

# number of features
size = 12 # end
# configure this parameter to 3 or 6 degrees of freedom
angles = 6 
# if you want to perform cross validation set "cross_val" to 1
cross_val = 0

#NN_model.save_weights(r'.\FINAL RESULTS\3DOF\dataset_with_constraints_no6_3WRIST_15K.h5')
#np.save(r'.\FINAL RESULTS\3DOF\dataset_with_constraints_no6_3WRIST_15K_accuracies_cross_val.npy', accuracies)

if cross_val == 1:
    
    df = df.drop(['Unnamed: 0'], axis = 1)
    
    x_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    y_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    
    X = df.iloc[:,:size]; 
    y = df.iloc[:,size:]; 
    
    X_s = x_scaler.fit_transform(X)
    y_s = y_scaler.fit_transform(y)
    
    #X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size = 0.2)
    
    #           cross-validation
    
    from sklearn.model_selection import KFold
    
    kfold = KFold(5, True, 1)
    count = 0
    accuracies = []
        
    for train, test in kfold.split(df):
    #	print('train: %s, test: %s' % (df[train], df[test]))
        count = count +1
        print("K_FOLD ITERATION NUMBER ",count)
        X_train = X_s[train[0]:train[-1],:]
        X_test = X_s[test[0]:test[-1],:]
        y_train = y_s[train[0]:train[-1]]
        y_test = y_s[test[0]:test[-1]] 
        
    
        #           Sequential network start
        
        NN_model = Sequential()
        
        # The Input Layer :
        NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
        
        # The Hidden Layers :
        NN_model.add(Dense(100, kernel_initializer='normal',activation='relu'))
        #NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
        NN_model.add(Dense(100, kernel_initializer='normal',activation='relu'))
        
        # The Output Layer :
        NN_model.add(Dense(angles, kernel_initializer='normal',activation='linear'))
        
        NN_model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['accuracy'])
        
        NN_model.summary()
        early_stop = EarlyStopping(monitor='acc', patience=15)
        
        m=NN_model.fit(X_train, y_train, epochs=300, validation_split = 0.2, callbacks=[early_stop])
       
        
        #           Sequential network end
        
        # Get training and test loss histories
        training_loss = m.history['loss']
        test_loss = m.history['val_loss']
        
        # Get training and test accuracy histories
        training_acc = m.history['accuracy']
        test_acc = m.history['val_accuracy']
        
        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        
        # Visualize loss history
        plt.figure()
        plt.title('Loss')
        plt.plot(epoch_count, training_loss)
        plt.plot(epoch_count, test_loss)
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.show()
        
        # Visualize accuracy history
        plt.figure()
        plt.title('Acuracy')
        plt.plot(epoch_count, training_acc)
        plt.plot(epoch_count, test_acc)
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy value')
        plt.show()
        
        
        y_pred = NN_model.predict(X_test)
        
        y_pred = pd.DataFrame(y_scaler.inverse_transform(y_pred))
        y_test = pd.DataFrame(y_scaler.inverse_transform(y_test))
        
        
        result_mse = []
        result_mae = []
        
        for i in range(angles):
        
            mse = mean_squared_error(y_test.iloc[:,i], y_pred.iloc[:,i])
            rmse = sqrt(mse) 
            mae = mean_absolute_error(y_test.iloc[:,i], y_pred.iloc[:,i])
            result_mse.append(mse)
            result_mae.append(mae)
        
        print("RMSE", result_mse)
        print("MAE", result_mae)
        
    #    y_pred = pd.DataFrame(y_pred)
    #    X_test = pd.DataFrame(x_scaler.inverse_transform(X_test))
    #    X_test = pd.DataFrame(X_test)
    #    
    #    y_pred.to_csv(r'.\y_pred.csv')
    #    y_test.to_csv(r'.\y_test.csv')
    #    X_test.to_csv(r'.\X_test.csv')
    
        accuracies.append(result_mae)
    
    mean_acc = []
    std_acc = []
    accuracies = np.array(accuracies)
    for i in range(len(accuracies[0])):
        mean_acc.append(np.mean(accuracies[:,i]))
        std_acc.append(np.std(accuracies[:,i]))
    print(mean_acc)
    print(std_acc)
  
    
    
else:
    
    df = df.drop(['Unnamed: 0'], axis = 1)
    #df = df.iloc[0:1500,:]
    
    x_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    y_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    
    X = df.iloc[:,:size]; 
    y = df.iloc[:,size:]; 
    
    X_s = x_scaler.fit_transform(X)
    y_s = y_scaler.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size = 0.2)
    
                
    #          Start of sequential neural network
    
    NN_model = Sequential()
    
    # Input layer
    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
    
    # Hidden layers
    NN_model.add(Dense(100, kernel_initializer='normal',activation='relu'))
    #NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(100, kernel_initializer='normal',activation='relu'))
    
    # Output layer
    NN_model.add(Dense(angles, kernel_initializer='normal',activation='linear'))
    
    NN_model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['accuracy'])
    
    NN_model.summary()
    early_stop = EarlyStopping(monitor='acc', patience=15)
    
    m=NN_model.fit(X_train, y_train, epochs=300, validation_split = 0.2, callbacks=[early_stop])
    
    #           Sequential neural network end
    
    
    # Get training and test loss histories
    training_loss = m.history['loss']
    test_loss = m.history['val_loss']
    
    # Get training and test accuracy histories
    training_acc = m.history['accuracy']
    test_acc = m.history['val_accuracy']
    
    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    
    # Visualize loss history
    plt.figure()
    plt.title('Loss')
    plt.plot(epoch_count, training_loss)
    plt.plot(epoch_count, test_loss)
    plt.legend(['Train', 'Test'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.show()
    
    # Visualize accuracy history
    plt.figure()
    plt.title('Acuracy')
    plt.plot(epoch_count, training_acc)
    plt.plot(epoch_count, test_acc)
    plt.legend(['Train', 'Test'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy value')
    plt.show()

    y_pred = NN_model.predict(X_test)
 
    y_pred = pd.DataFrame(y_scaler.inverse_transform(y_pred))
    y_test = pd.DataFrame(y_scaler.inverse_transform(y_test))
 
    result_mse = []
    result_mae = []
    
    for i in range(angles):
    
        mse = mean_squared_error(y_test.iloc[:,i], y_pred.iloc[:,i])
        rmse = sqrt(mse) 
        mae = mean_absolute_error(y_test.iloc[:,i], y_pred.iloc[:,i])
        result_mse.append(mse)
        result_mae.append(mae)
    
    print("RMSE", result_mse)
    print("MAE", result_mae)
    
    # y_pred = pd.DataFrame(y_pred)
    # X_test = pd.DataFrame(x_scaler.inverse_transform(X_test))
    # X_test = pd.DataFrame(X_test)
    
    # y_pred.to_csv(r'.\y_pred.csv')
    # y_test.to_csv(r'.\y_test.csv')
    # X_test.to_csv(r'.\X_test.csv')
