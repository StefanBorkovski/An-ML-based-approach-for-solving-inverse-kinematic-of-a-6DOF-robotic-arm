# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:23:34 2019

@author: Stefan Borkovski
"""

# This code is used for creating data set for 6DOF robotic arm with use of direct kinematic.

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
    Ta_b = Ta_b.subs(s)  

    # Substitute in the DH parameters     
    
    return Ta_b

def calculate_position(teta1, teta2, teta3, teta4, teta5, teta6):    

     
    theta1, theta2, theta3, theta4, theta5, theta6 = symbols('theta1:7')
    alpha0, alpha1, alpha2, alpha3, alpha4, alpha5 = symbols('alpha0:6')
    d1, d2, d3, d4, d5, d6 = symbols('d1:7')    
    a0, a1, a2, a3, a4, a5 = symbols('a0:6')

    #  DH parameters

    kuka_s = {alpha0:   -pi/2,  d1:  0.675,      a0:   0.260,
              alpha1:   0,      d2:     0,       a1:   0.68,    
              alpha2:   pi/2,   d3:     0,       a2:   0,       theta2: (theta2 - pi/2),
              alpha3:  -pi/2,   d4:  -0.67,      a3:   0,
              alpha4:   pi/2,   d5:     0,       a4:   0,
              alpha5:   pi,     d6:     -0.158,  a5:   0, }              

    # Define Modified DH Transformation matrix
            
    T0_1 = build_mod_dh_matrix(s=kuka_s, theta=theta1, alpha=alpha0, d=d1, a=a0)
    T1_2 = build_mod_dh_matrix(s=kuka_s, theta=theta2, alpha=alpha1, d=d2, a=a1)
    T2_3 = build_mod_dh_matrix(s=kuka_s, theta=theta3, alpha=alpha2, d=d3, a=a2)
    T3_4 = build_mod_dh_matrix(s=kuka_s, theta=theta4, alpha=alpha3, d=d4, a=a3)
    T4_5 = build_mod_dh_matrix(s=kuka_s, theta=theta5, alpha=alpha4, d=d5, a=a4)
    T5_6 = build_mod_dh_matrix(s=kuka_s, theta=theta6, alpha=alpha5, d=d6, a=a5)    

    # Create individual transformation matrices

    T0_2 = simplify(T0_1 * T1_2)    
    T0_3 = simplify(T0_2 * T2_3)    
    T0_4 = simplify(T0_3 * T3_4)   
    T0_5 = simplify(T0_4 * T4_5)    
    T0_G = simplify(T0_5 * T5_6)  
        
    T_total = simplify( T0_G )    
    result = T_total.evalf(subs={theta1: teta1, theta2: teta2, theta3: teta3, theta4: teta4,
                                 theta5: teta5, theta6: teta5})    
    final = np.array(result).astype(np.float64)    
    return final

#       creating data set

start_time = time.time()

# MORE TIGHT constraints for my configuration no6

# defining the constraints for every joint
number_points = 2 #1000

angle0 = [random.uniform(-1.57, 1.57) for i in range(0,number_points)]    # 90-90 degrees
angle1 = [random.uniform(-1.05, 1.05) for i in range(0,number_points)]    # 60-60 degrees
angle2 = [random.uniform(-1.05, 1.05) for i in range(0,number_points)]    # 60-60 degrees
angle3 = [random.uniform(-0.78, 0.78) for i in range(0,number_points)]    # 45-45 degrees
angle4 = [random.uniform(-0.78, 0.78) for i in range(0,number_points)]    # 45-45 degrees
angle5 = [random.uniform(-0.78, 0.78) for i in range(0,number_points)]    # 45-45 degrees

angles = np.zeros([1,6],dtype=float)
positions = np.zeros([1,3],dtype=float)
n = np.zeros([1,3],dtype=float)
o = np.zeros([1,3],dtype=float)
a = np.zeros([1,3],dtype=float)

# calculation of the final transform matrix and generation of dataset

for i in range(0,number_points):
    
    print(i)
    ang = []
    ang.append( [math.degrees(angle0[i]), math.degrees(angle1[i]), math.degrees(angle2[i]), 
                 math.degrees(angle3[i]), math.degrees(angle4[i]), math.degrees(angle5[i])] )
    ang = np.asarray(ang)    
    result = calculate_position(angle0[i], angle1[i], angle2[i], angle3[i], angle4[i], angle5[i])
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

df = pd.DataFrame(np.concatenate((positions,n,o,a,angles), axis = 1) )
df = df.iloc[1:]
# df.to_csv(r'.\codes_and_data\no.<1>6DOF.csv')

end_time = time.time()
time = end_time - start_time

print('Time needed',time)
print(df)
    
# needed 15 sec    
        