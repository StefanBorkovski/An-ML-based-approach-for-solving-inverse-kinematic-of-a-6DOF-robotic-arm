# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:00:10 2019

@author: Stefan Borkovski
"""

import pandas as pd
import numpy as np

# This code is used for merging two data sets. Also for saving the new created dataset.

path = r'.\datasets_6DOF'

df0 = pd.read_csv(path + r'\dataset_myplace1_with_constraints_no6.csv',  encoding = 'utf8')
df0 = df0.drop(['Unnamed: 0'], axis = 1)

df1 = pd.read_csv(path+ r'\dataset_myplace2_with_constraints_no6.csv',  encoding = 'utf8')
df1 = df1.drop(['Unnamed: 0'], axis = 1)

df2 = pd.read_csv(path+ r'\dataset_myplace3_with_constraints_no6.csv',  encoding = 'utf8')
df2 = df2.drop(['Unnamed: 0'], axis = 1)

#df0 = df0.iloc[:,1:]
#df1 = df1.iloc[:,1:]
#df2 = df1.iloc[:,1:]

df = np.concatenate((df0, df1, df2), axis = 0)
df = pd.DataFrame(df)

# df.to_csv(r'.\datasets_07.08.2019_with_constraints_no6-7\dataset_myplace_with_constraints_no6_merged.csv')