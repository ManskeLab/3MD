#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#---------------------------------------------
# File: projectPercent.py
#
# Created By: dakondro
# Created On: Sat Jan 18 13:50:55 2020
#
# Modified By: Michael Kuczynski
# Modified On: Fri Jul 09 2021
# Modification Notes:
#       -Code cleanup and comments
#---------------------------------------------
# Description:
#
#
#---------------------------------------------

import numpy as np

def project_precent_a_array(point_x,point_y,a_x,a_y,b_x,b_y):
    #Has errors!!!!!!
    #We find the line between the two input points
    #point_x=float(point_x)
    m=(b_y-a_y)/(b_x-a_x)
    b=b_y-m*b_x
    #Our projection must be an orthogonal line to this
    m_orthogonal=-1/m
    b_orthogonal=point_y-m_orthogonal*point_x
    #Find where they intercept
    x_intercept=(b_orthogonal-b)/(m-m_orthogonal)
    y_intercept=m_orthogonal*x_intercept+b_orthogonal
    #Find the distances
    a_2_b=np.sqrt((a_x-b_x)**2+(a_y-b_y)**2)
    b_2_intercept=np.sqrt((b_x-x_intercept)**2+(b_y-y_intercept)**2)
    a_2_intercept=np.sqrt((a_x-x_intercept)**2+(a_y-y_intercept)**2)
    percent_a=b_2_intercept/a_2_b
    #If our x point is to the right point b and the slope is positive and point a is the lower point set to 1
    #percent_a[((b_x-a_x)*b*(x_intercept-a_x))<0]=1
    percent_a[b_2_intercept>a_2_b]=1 #Only works if the distance is not outside of length from a to b
    #print(np.sum(a_2_intercept>a_2_b))
    percent_a[a_2_intercept>a_2_b]=0
    if (np.sum(percent_a > 1))>0:
        print('There are some precentages greater than 1')
    #Not needed
    #percent_a[percent_a>1]=1
    #percent_a[percent_a<0]=0

    return percent_a


def extend_x_y(a_x,a_y,b_x,b_y,extension=1):
    #point x and point y are arrays
    c_x=(b_x-a_x)*(extension+1)+a_x
    c_y=(b_y-a_y)*(extension+1)+a_y
    
    return c_x,c_y


def project_precent_a_array_old(point_x,point_y,a_x,a_y,b_x,b_y):
    #Not the best but it works... 
    #point x and point y are arrays
    point_2_a=np.sqrt((point_x-a_x)**2+(point_y-a_y)**2)
    point_2_b=np.sqrt((point_x-b_x)**2+(point_y-b_y)**2)
    a_2_b=np.sqrt((a_x-b_x)**2+(a_y-b_y)**2)

    s=(point_2_a+point_2_b+a_2_b)/2 #Herons Formula to find area
    A=np.sqrt(s*(s-point_2_a)*(s-point_2_b)*(s-a_2_b)) #Herons Formula to find area
    h=2*A/a_2_b  #1/2 base*height = Area
    project_2_a=np.sqrt(point_2_a**2-h**2)
    project_2_b=np.sqrt(point_2_b**2-h**2)
    percent_a=project_2_b/a_2_b
    #percent_b=project_2_a/a_2_b
    percent_a[percent_a>1]=1
    percent_a[percent_a<0]=0
    #percent_a[(project_2_b+project_2_a>a_2_b) & (project_2_b>project_2_a)]=1
    #percent_a[(project_2_b+project_2_a>a_2_b) & (project_2_a>project_2_b)]=0
    #print(np.max(percent_a))
        
    return percent_a
