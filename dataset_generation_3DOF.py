# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:23:34 2019

@author: Stefan Borkovski
"""

# This code is used for creating data set for 3DOF robotic arm with use of direct kinematic.

from sympy import symbols, pi, sin, cos, simplify
from sympy.matrices import Matrix
import numpy as np
import random 
import matplotlib.pyplot as plt
import pandas as pd
import time
import math

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
        
    result = T_total.evalf(subs={theta1: teta1, theta2: teta2, theta3: teta3})
    
    final = np.array(result).astype(np.float64)
    
    return final

#       creating data set

start_time = time.time()

number_of_points = 2 #160000

# constraints 3 wrists no6

angle0 = [random.uniform(-1.57, 1.57) for i in range(0,number_of_points)]    # 90-90 degrees
angle1 = [random.uniform(-1.05, 1.05) for i in range(0,number_of_points)]    # 60-60 degrees
angle2 = [random.uniform(-1.05, 1.05) for i in range(0,number_of_points)]    # 60-60 degrees


angles = np.zeros([1,3],dtype=float)
positions = np.zeros([1,3],dtype=float)
n = np.zeros([1,3],dtype=float)
o = np.zeros([1,3],dtype=float)
a = np.zeros([1,3],dtype=float)

for i in range(0, number_of_points):
    print(i)
    ang = []
    ang.append( [math.degrees(angle0[i]), math.degrees(angle1[i]), math.degrees(angle2[i])] )

    ang = np.asarray(ang)
    
    result = calculate_position(angle0[i], angle1[i], angle2[i])
    print(result)

    position_xyz = [] 
    position_xyz.append( [result[0][3], result[1][3], result[2][3]] )
    position_xyz = np.asarray(position_xyz)
    
    n_xyz = []
    n_xyz.append( [result[0][0], result[1][0], result[2][0]] )
    
    o_xyz = []
    o_xyz.append( [result[0][1], result[1][1], result[2][1]] )
    
    a_xyz = []
    a_xyz.append( [result[0][2], result[1][2], result[2][2]] )
    
    angles = np.concatenate((angles,ang))
    positions = np.concatenate((positions,position_xyz))
    n = np.concatenate((n, n_xyz))
    o = np.concatenate((o, o_xyz))
    a = np.concatenate((a, a_xyz))
    
#    print('Positions x, y, z', position_xyz)

df = pd.DataFrame(np.concatenate((positions,n,o,a,angles), axis = 1) )
df = df.iloc[1:]
# df.to_csv(r'.\codes_and_data\no.<1>3DOF.csv')

end_time = time.time()
time = end_time - start_time

print('Time needed',time)

    
        