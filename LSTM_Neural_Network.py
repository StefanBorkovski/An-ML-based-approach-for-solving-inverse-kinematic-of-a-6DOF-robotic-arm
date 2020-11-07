# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:07:37 2019

@author: Stefan Borkovski
"""

# This code is used for training and testing the LSTM neural network model for 3 and 6 DOF 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model

# loading the data sets

###          3DOF 

#df = pd.read_csv(r'.\dataset_with_constraints_no6_3WRIST_15K.csv',  encoding = 'utf8')
#  MAE [0.10624026581956693, 0.15616805447615276, 0.13938704291074547]
#  cross_val MAE [0.16358137192021718, 0.1487296730002861, 0.15814576777707862]

###          6DOF (dataset with constraints no 6/7)

df = pd.read_csv(r'.\datasets_6DOF\dataset_myplace_with_constraints_no6_merged_plus3.csv',  encoding = 'utf8')
# MAE [1.327897870427836, 0.6940683958030475, 0.9348772063292176, 1.1812736815637395, 1.1281554071974593, 22.541354624604665]
# cross_val [0.8796283584810153, 0.5836110064051009, 0.5461870389664586, 0.9794998462462202, 0.5772626092840545, 22.193187113077187]

df = df.drop(['Unnamed: 0'], axis = 1)

# number of features
size = 12 # end
# configure this parameter to 3 or 6 degrees of freedom
angles = 6 # 
# if you want to perform cross validation set "cross_val" to 1
cross_val = 1

if cross_val == 0 :
    
    x_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    y_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    
    X = df.iloc[:,:size]; 
    y = df.iloc[:,size:]; 
    
    X_s = x_scaler.fit_transform(X)
    y_s = y_scaler.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2)
    
    X_train = pd.DataFrame(X_train); X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train); y_test = pd.DataFrame(y_test)
    
    
    #           Start of LSTM
    
    X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]) )
    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]) )
    
    model = Sequential()
    
    model.add(LSTM(64, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(angles, activation='linear'))
    
    callbacks = [EarlyStopping(monitor='val_acc', patience = 10)]
    
    model.compile(
        loss='mae',
        optimizer='Adam',
        metrics=['accuracy'])
    
    model_f=model.fit(X_train,
              y_train,
              callbacks = callbacks,
              epochs=100,
              validation_split = 0.2)
    
    y_pred = model.predict(X_test)

# load pretrained models for testing
    
#    model = load_model(r'.\FINAL RESULTS\LSTM\6DOF\dataset_myplace_with_constraints_no6_merged_plus3_LSTM.h5')
#    model.save(r'.\FINAL RESULTS\LSTM\3DOF\dataset_with_constraints_no6_3WRIST_15K.h5')
    
    #           LSTM end
        
    # Get training and test loss histories
    training_loss = model_f.history['loss']
    test_loss = model_f.history['val_loss']
    
    # Get training and test accuracy histories
    training_acc = model_f.history['accuracy']
    test_acc = model_f.history['val_accuracy']
    
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
    
    
    X_test_r = pd.DataFrame(x_scaler.inverse_transform( pd.DataFrame(np.reshape(X_test,(X_test.shape[0],X_test.shape[2]))) ))
    
    # y_pred.to_csv(r'.\y_pred.csv')
    # y_test.to_csv(r'.\y_test.csv')
    # X_test_r.to_csv(r'.\X_test.csv')

else:
    
    scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    
    X = df.iloc[:,:size]; 
    y = df.iloc[:,size:]; 
    
    X_scaler = scaler.fit(X)
    X_s = scaler.transform(X)
#    X_s = pd.DataFrame(X)
    
    y_scaler = scaler.fit(y)
    y_s = scaler.transform(y)
#    y_s = pd.DataFrame(y)
    
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
        #           cross-validation
    
    
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
    
        #           LSTM start
        
        X_test = pd.DataFrame(X_test)
        X_train = pd.DataFrame(X_train)
        
        X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]) )
        X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]) )
        
        model = Sequential()
        
        model.add(LSTM(64, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
        #model.add(Dropout(0.1))
        model.add(LSTM(128, activation='relu', return_sequences=True))
        #model.add(Dropout(0.1))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(LSTM(256, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(angles, activation='linear'))
        
        callbacks = [EarlyStopping(monitor='val_acc', patience = 10)]
        
        model.compile(
            loss='mae',  # dont change it
            optimizer='Adam',
            metrics=['accuracy'])
        
        model_f=model.fit(X_train,
                  y_train,
                  callbacks = callbacks,
                  epochs=100,
                  validation_split = 0.2)
        
        y_pred = model.predict(X_test)
        
        #model = load_model(r'.\datasets_05.08.2019_with_sonstraints_no2-3-4\61223no4.h5')
        #model.save(r'.\datasets_07.08.2019_with_constraints_no6-7\dataset_with_constraints_no6_3WRIST_15K_LSTM.h5')
        
        #           LSTM end
        
        
        # Get training and test loss histories
        training_loss = model_f.history['loss']
        test_loss = model_f.history['val_loss']
        
        # Get training and test accuracy histories
        training_acc = model_f.history['accuracy']
        test_acc = model_f.history['val_accuracy']
        
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
        
        #print("RMSE", result_mse)
        print("MAE", result_mae)
        
        accuracies.append(result_mae)
    
    mean_acc = []
    std_acc = []
    accuracies = np.array(accuracies)
    for i in range(len(accuracies[0])):
        mean_acc.append(np.mean(accuracies[:,i]))
        std_acc.append(np.std(accuracies[:,i]))
    print('mean', mean_acc)
    print('std', std_acc)