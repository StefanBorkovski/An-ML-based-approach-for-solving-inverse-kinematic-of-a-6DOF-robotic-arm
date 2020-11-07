# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:38:14 2019

@author: Stefan Borkovski
"""

# This code is used to vizualize the robot workspace (the dataset).

from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Reading the data set
df = pd.read_csv(r'.\datasets_6DOF\dataset_myplace_with_constraints_no6_merged_plus3.csv',  encoding = 'utf8')

# Setting number of points for vizualization
number_points = 1000

df = df.iloc[0:number_points,:]
df = df.iloc[:,1:]

xyz = df.iloc[:,0:3]

# Plotting the points
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xyz.iloc[:,0]*10, xyz.iloc[:,1], xyz.iloc[:,2], c=xyz.iloc[:,2]);

