# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 23:31:01 2019

@author: Stefan Borkovski
"""

# This code is used to clear the dataset from duplicates (points that are very close). This method 
# proved that it is improving the training phase. 

import time
import pandas as pd
import numpy as np

df = pd.read_csv(r'.\datasets_6DOF\dataset_myplace_with_constraints_no6_merged_plus3.csv',  encoding = 'utf8')
df = df.drop(['Unnamed: 0'], axis = 1)

st=time.time()

counter = 0
decimal = 1
list_duplicates = []
drop_list = []
for i in range(len(df)):
    x_comp = round(df.iloc[i][0],decimal)
    y_comp = round(df.iloc[i][1],decimal)
    z_comp = round(df.iloc[i][2],decimal)
    for j in range(len(df)):
        flag = False
        if i != j:
            if ( ( round(df.iloc[j][0],decimal) == x_comp ) & ( round(df.iloc[j][1],decimal) == y_comp ) & ( round(df.iloc[j][2],decimal) == z_comp ) ) :
                counter = counter +1
                for k in range(len(list_duplicates)):
                    if (list_duplicates[k] == (j,i)) :
                        flag = True
                if flag == False :
                    list_duplicates.append((i,j))  
                    drop_list.append(i)
    print(i)
print(counter)
df = df.drop(drop_list, axis=0)
df = df.reset_index(drop=True)
en = time.time()
print('Time needed',en-st)
# df.to_csv(r'.\datasets_6DOF\dataset_myplace_with_constraints_no6_merged_plus3_removed_duplicates.csv')