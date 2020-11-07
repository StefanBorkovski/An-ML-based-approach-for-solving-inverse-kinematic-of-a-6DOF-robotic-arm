# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:31:33 2019

@author: Stefan Borkovski
"""

### Forward kinematic - Denavit Hartenberg (DH) parameters - Test

# This code is used to generate final transformation matrix using DH Convention


from sympy import symbols, pi, sin, cos, simplify
from sympy.matrices import Matrix


def build_mod_dh_matrix(s, theta, alpha, d, a):

    Ta_b = Matrix([ [cos(theta), -cos(alpha)*sin(theta),  sin(alpha)*sin(theta), a*cos(theta)],
                    [sin(theta),  cos(alpha)*cos(theta), -sin(alpha)*cos(theta), a*sin(theta)],
                    [0,           sin(alpha),             cos(alpha),            d           ],
                    [0,           0,                      0,                     1]          ])
    
    # Substitute in the DH parameters 
    
    Ta_b = Ta_b.subs(s)
    
    return Ta_b

# DH param symbol
        
theta1, theta2, theta3, theta4, theta5, theta6 = symbols('theta1:7')
alpha0, alpha1, alpha2, alpha3, alpha4, alpha5 = symbols('alpha0:6')
d1, d2, d3, d4, d5, d6 = symbols('d1:7')    # link offsets
a0, a1, a2, a3, a4, a5 = symbols('a0:6')    # link lengths

#  DH parameters, second approach

kuka_s = {alpha0:   -pi/2,  d1:  0.675,      a0:   0.260,
          alpha1:   0,     d2:     0,       a1:   0.68,   
          alpha2:   pi/2,  d3:     0,       a2:   0,    theta2: (theta2 - pi/2),
          alpha3:  -pi/2,  d4:  -0.67,       a3:   0,
          alpha4:   pi/2,  d5:     0,       a4:   0,
          alpha5:   pi,     d6:     -0.158,   a5:   0, }
          

# Define Modified DH Transformation matrix
          
T0_1 = build_mod_dh_matrix(s=kuka_s, theta=theta1, alpha=alpha0, d=d1, a=a0)
T1_2 = build_mod_dh_matrix(s=kuka_s, theta=theta2, alpha=alpha1, d=d2, a=a1)
T2_3 = build_mod_dh_matrix(s=kuka_s, theta=theta3, alpha=alpha2, d=d3, a=a2)
T3_4 = build_mod_dh_matrix(s=kuka_s, theta=theta4, alpha=alpha3, d=d4, a=a3)
T4_5 = build_mod_dh_matrix(s=kuka_s, theta=theta5, alpha=alpha4, d=d5, a=a4)
T5_6 = build_mod_dh_matrix(s=kuka_s, theta=theta6, alpha=alpha5, d=d6, a=a5)

# Create individual transformation matrices

T0_2 = simplify(T0_1 * T1_2)    # base link to link 2
T0_3 = simplify(T0_2 * T2_3)    # base link to link 3
T0_4 = simplify(T0_3 * T3_4)    # base link to link 4
T0_5 = simplify(T0_4 * T4_5)    # base link to link 5
T0_G = simplify(T0_5 * T5_6)    # base link to link 6

T_total = simplify( T0_G )

result = T_total.evalf(subs={theta1: 0, theta2: 0, theta3: 0, theta4: 0, theta5: 0, theta6: 0})

print(result)