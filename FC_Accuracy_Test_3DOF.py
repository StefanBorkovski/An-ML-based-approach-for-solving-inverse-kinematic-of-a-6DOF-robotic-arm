# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:17:52 2019

@author: Stefan Borkovski
"""

# This code is used for 3D visualization of training and test points for robotic arm with 3 DOF.

from sympy import symbols, pi, sin, cos, simplify
from sympy.matrices import Matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import open3d as o3d
import math
from mpl_toolkits.mplot3d import Axes3D
import time

def build_mod_dh_matrix(s, theta, alpha, d, a):

    # transformation matrix  
    
    Ta_b = Matrix([ [cos(theta), -cos(alpha)*sin(theta),  sin(alpha)*sin(theta), a*cos(theta)],
                    [sin(theta),  cos(alpha)*cos(theta), -sin(alpha)*cos(theta), a*sin(theta)],
                    [0,           sin(alpha),             cos(alpha),            d           ],
                    [0,           0,                      0,                     1]          ])
    
    # Substitute in the DH parameters 
    
    Ta_b = Ta_b.subs(s)
    
    return Ta_b

def calculate_position(teta1, teta2, teta3):
    
    # DH param symbol
        
    theta1, theta2, theta3 = symbols('theta1:4')
    alpha0, alpha1, alpha2 = symbols('alpha0:3')
    d1, d2, d3 = symbols('d1:4')    # link offsets
    a0, a1, a2 = symbols('a0:3')    # link lengths
        
    #  DH parameters
    
#    kuka_s = {alpha0:   -pi/2,  d1:  0.675,      a0:   0.26,  
#              alpha1:   0,      d2:     0,       a1:   0.68,   theta2: (theta2 - pi/2), 
#              alpha2:   0,      d3:     0,       a2:   0.67,  }  
        
    kuka_s = {alpha0:   -pi/2,  d1:  0.675,      a0:   0.26,  
              alpha1:   0,      d2:     0,       a1:   0.68,  
              alpha2:   0,      d3:     0,       a2:   0.67,  }                
    
    # Define Modified DH Transformation matrix
              
    T0_1 = build_mod_dh_matrix(s=kuka_s, theta=theta1, alpha=alpha0, d=d1, a=a0)
    T1_2 = build_mod_dh_matrix(s=kuka_s, theta=theta2, alpha=alpha1, d=d2, a=a1)
    T2_3 = build_mod_dh_matrix(s=kuka_s, theta=theta3, alpha=alpha2, d=d3, a=a2)

    
    # Create individual transformation matrices
    
    T0_2 = simplify(T0_1 * T1_2)    # base link to link 2
    T0_3 = simplify(T0_2 * T2_3)    # base link to link 3

    
    
    # Correction to account for orientation difference between the gripper and
    #   the arm base (rotation around Z axis by 180 deg and Y axis by -90 deg)
    
#    R_z = Matrix([[     cos(pi), -sin(pi),          0, 0],
#                  [     sin(pi),  cos(pi),          0, 0],
#                  [           0,        0,          1, 0],
#                  [           0,        0,          0, 1]])
#        
#    R_y = Matrix([[  cos(-pi/2),        0, sin(-pi/2), 0],
#                  [           0,        1,          0, 0],
#                  [ -sin(-pi/2),        0, cos(-pi/2), 0],
#                  [           0,        0,          0, 1]])
#        
#    R_corr = simplify(R_y * R_z)
    
    ## Total homogeneous transform between base and gripper with orientation correction applied
    
    T_total = simplify( T0_3 )
    
    # Numerically evaluate transforms (compare this with output of tf_echo)
    
#    print(T0_1)
#    print(T1_2)
#    print(T2_3)
#    print(T3_4)
#    print(T4_5)
#    print(T5_6)
    
    #print(T0_G)
    #print('T0_1 = ', T0_1.evalf(subs={theta1: 0, theta2: 0, theta3: 0, theta4: 0, theta5: 0, theta6:0}))
    #print('T1_2 = ', T0_2.evalf(subs={theta1: 0, theta2: 0, theta3: 0, theta4: 0, theta5: 0, theta6:0}))
    #print('T2_3 = ', T0_3.evalf(subs={theta1: 0, theta2: 0, theta3: 0, theta4: 0, theta5: 0, theta6:0}))
    #print('T3_4 = ', T0_4.evalf(subs={theta1: 0, theta2: 0, theta3: 0, theta4: 0, theta5: 0, theta6:0}))
    #print('T4_5 = ', T0_5.evalf(subs={theta1: 0, theta2: 0, theta3: 0, theta4: 0, theta5: 0, theta6:0}))
    #print('T5_6 = ', T0_6.evalf(subs={theta1: 0, theta2: 0, theta3: 0, theta4: 0, theta5: 0, theta6:0}))
    #print('T0_G = ', T0_G.evalf(subs={theta1: 0, theta2: 0, theta3: 0, theta4: 0, theta5: 0, theta6:0}))
    
    result = T_total.evalf(subs={theta1: teta1, theta2: teta2, theta3: teta3})
    
    final = np.array(result).astype(np.float64)
    
    return final

path = r'.\results\sequential neural network\3DOF'
size = 10 #1000

y_pred = pd.read_csv(path + '\y_pred.csv',  encoding = 'utf8')
y_pred = y_pred.drop(['Unnamed: 0'], axis = 1)
y_pred = y_pred.iloc[0:size,:]

y_test = pd.read_csv(path + '\y_test.csv',  encoding = 'utf8')
y_test = y_test.drop(['Unnamed: 0'], axis = 1)
y_test = y_test.iloc[0:size,:]

X_test = pd.read_csv(path + '\X_test.csv',  encoding = 'utf8')
X_test = X_test.drop(['Unnamed: 0'], axis = 1)
X_test = X_test.iloc[0:size,:]

y_pred = y_pred.values

n = np.zeros([1,3],dtype=int)
o = np.zeros([1,3],dtype=int)
a = np.zeros([1,3],dtype=int)
positions = np.zeros([1,3],dtype=int)
st = time.time()
for i in range(len(y_pred)):
    
    print(i)
    final = calculate_position(math.radians(y_pred[i][0]), math.radians(y_pred[i][1]), math.radians(y_pred[i][2]))
    
    position_xyz = []
    position_xyz.append( [final[0][3], final[1][3], final[2][3]] )

    n_xyz = []
    n_xyz.append( [final[0][0], final[1][0], final[2][0]] )
    
    o_xyz = []
    o_xyz.append( [final[0][1], final[1][1], final[2][1]] )
    
    a_xyz = []
    a_xyz.append( [final[0][2], final[1][2], final[2][2]] )
    
    positions = np.concatenate((positions,position_xyz) )
    n = np.concatenate((n, n_xyz))
    o = np.concatenate((o, o_xyz))
    a = np.concatenate((a, a_xyz))
    

X_pred = pd.DataFrame(np.concatenate((positions,n,o,a), axis = 1) )
X_pred = X_pred.iloc[1:,:]
X_pred.to_csv(path + '\X_pred.csv')
  
result_mse = []
result_mae = []

## only for plotting
#X_pred = pd.read_csv(path + '\X_pred.csv',  encoding = 'utf8')
#X_pred = X_pred.drop(['Unnamed: 0'], axis = 1)
#X_pred = X_pred.iloc[0:size,:]

X_pred = X_pred.iloc[:,:3]
X_test = X_test.iloc[:,:3]

X_pred_r = []
X_test_r = []

for i in range(len(X_pred)):
    
    X_pred_r.append( math.sqrt(X_pred.iloc[i][0]*X_pred.iloc[i][0] + X_pred.iloc[i][1]*X_pred.iloc[i][1] + X_pred.iloc[i][2]*X_pred.iloc[i][2]) )
    X_test_r.append( math.sqrt(X_test.iloc[i][0]*X_test.iloc[i][0] + X_test.iloc[i][1]*X_test.iloc[i][1] + X_test.iloc[i][2]*X_test.iloc[i][2]) )

X_pred['r'] = X_pred_r
X_test['r'] = X_test_r

for i in range(4):

   mse = mean_squared_error(X_test.iloc[:,i], X_pred.iloc[:,i])
   rmse = sqrt(mse) 
   mae = mean_absolute_error(X_test.iloc[:,i], X_pred.iloc[:,i])
   result_mse.append(mse)
   result_mae.append(mae)

print("RMSE", result_mse)
print("MAE", result_mae)

en = time.time()
print("time needed",en-st)

y_pred = pd.DataFrame(y_pred)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_pred.iloc[:100,0], X_pred.iloc[:100,1], X_pred.iloc[:100,2], color='r')
ax.scatter3D(X_test.iloc[:100,0], X_test.iloc[:100,1], X_test.iloc[:100,2], color='g')