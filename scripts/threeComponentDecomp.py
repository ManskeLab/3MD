#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#---------------------------------------------
# File: threeComponentDecomp.py
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

import os
import csv
import numpy as np

from PIL import Image

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from projectPercent import project_precent_a_array, extend_x_y


def Component3_Decomp(dicomPath_50keV, dicomPath_65keV, Patient, Leg, filterimage=True, Correct_bone=True, Extend_seed=True):                
    # outpath='C:/Users/dakondro.UC/Documents/Gout-conv/Written.nii'
    parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outpath = os.path.join(parentDir, 'output')

    #Load the two images I expect that they are dicoms
    reader50keV = vtk.vtkDICOMImageReader() 
    reader50keV.SetDirectoryName(dicomPath_50keV) 
    reader50keV.SetDataScalarTypeToUnsignedShort()
    reader50keV.UpdateWholeExtent()
    reader50keV.Update()
    
    reader65keV = vtk.vtkDICOMImageReader() 
    reader65keV.SetDirectoryName(dicomPath_65keV) 
    reader65keV.SetDataScalarTypeToUnsignedShort()
    reader65keV.UpdateWholeExtent()
    reader65keV.Update()
    
    (xsize, ysize, zsize) = reader50keV.GetOutput().GetDimensions()
    (xsize, ysize, zsize) = reader50keV.GetOutput().GetDimensions()

    
    #print('Should we filter?')
    if filterimage:
        #print('Applying Gaussian Smooth') 
        filter_image1 = vtk.vtkImageGaussianSmooth()
        filter_image1.SetInputConnection(reader50keV.GetOutputPort())
        filter_image1.SetStandardDeviation(2)
        filter_image1.SetRadiusFactors(3,3,3)
        filter_image1.SetDimensionality(3)
        filter_image1.Update()
        nodes_vtk_array_50 = filter_image1.GetOutput().GetPointData().GetArray(0)

        filter_image.SetInputConnection(reader65keV.GetOutputPort())
        filter_image.Update()
        nodes_vtk_array_65 = filter_image.GetOutput().GetPointData().GetArray(0)
    else:
        #print('No Filter Applied')
        nodes_vtk_array_50 = reader50keV.GetOutput().GetPointData().GetScalars()
        nodes_vtk_array_65 = reader65keV.GetOutput().GetPointData().GetScalars()

        
    #image_array=np.zeros((int(xsize*ysize*zsize), 4))    
    image_array_50keV = vtk_to_numpy(nodes_vtk_array_50)
    image_array_65keV = vtk_to_numpy(nodes_vtk_array_65)
    
    Array_65keV = np.reshape(image_array_65keV, [xsize, ysize, zsize], order='F')
    Array_50keV = np.reshape(image_array_50keV, [xsize, ysize, zsize], order='F')
    Array_Dual = Array_65keV
    Array_Three = Array_65keV
    
    
    #-------------------------------------------------------
    # ROI Extration
    #--------------------------------------------------------
    square_size = 11
    ROI_list = []
    
    phantomROIsCSV = os.path.join(parentDir, 'Phantom_ROIs_Reformatted.csv')
    with open(phantomROIsCSV, newline = '') as ROIs:  # Why is this a txt file and not CSV?                                                                                        
        ROI_reader = csv.reader(ROIs, delimiter=',')
        for ROI in ROI_reader:
            ROI_list.append(ROI)
    ROI_array = np.array(ROI_list)
    
    ROI_array = ROI_array[1:,:] # slice out the top row (i.e., headers)
    ROI_array = ROI_array[ROI_array[:,0]==Patient,:] # Compares values, elementwise, between 'Patient' and first column of 'ROI_array'. Remove data for patients not in current 'Patient'
    ROI_array = ROI_array[ROI_array[:,1]==Leg,:] # Same as above but for right or left leg
    ROI_array = ROI_array[:,3:-1] # Slice out first two columns and last column
    ROI_array = ROI_array.astype(float)
    
    MSU_0_65keV = np.mean(Array_65keV[int(ROI_array[0,3]-square_size/2):int(ROI_array[0,3]+square_size/2), \
                                        int(ROI_array[1,3]-square_size/2):int(ROI_array[1,3]+square_size/2), \
                                        int(ROI_array[2,3]-square_size/2):int(ROI_array[2,3]+square_size/2)])
    MSU_0_50keV = np.mean(Array_50keV[int(ROI_array[0,3]-square_size/2):int(ROI_array[0,3]+square_size/2), \
                                        int(ROI_array[1,3]-square_size/2):int(ROI_array[1,3]+square_size/2), \
                                        int(ROI_array[2,3]-square_size/2):int(ROI_array[2,3]+square_size/2)])
    
    MSU_0625_65keV = np.mean(Array_65keV[int(ROI_array[0,4]-square_size/2):int(ROI_array[0,4]+square_size/2), \
                                        int(ROI_array[1,4]-square_size/2):int(ROI_array[1,4]+square_size/2), \
                                        int(ROI_array[2,4]-square_size/2):int(ROI_array[2,4]+square_size/2)])
    MSU_0625_50keV = np.mean(Array_50keV[int(ROI_array[0,4]-square_size/2):int(ROI_array[0,4]+square_size/2), \
                                        int(ROI_array[1,4]-square_size/2):int(ROI_array[1,4]+square_size/2), \
                                        int(ROI_array[2,4]-square_size/2):int(ROI_array[2,4]+square_size/2)])
    
    MSU_125_65keV = np.mean(Array_65keV[int(ROI_array[0,5]-square_size/2):int(ROI_array[0,5]+square_size/2), \
                                        int(ROI_array[1,5]-square_size/2):int(ROI_array[1,5]+square_size/2), \
                                        int(ROI_array[2,5]-square_size/2):int(ROI_array[2,5]+square_size/2)])
    MSU_125_50keV = np.mean(Array_50keV[int(ROI_array[0,5]-square_size/2):int(ROI_array[0,5]+square_size/2), \
                                        int(ROI_array[1,5]-square_size/2):int(ROI_array[1,5]+square_size/2), \
                                        int(ROI_array[2,5]-square_size/2):int(ROI_array[2,5]+square_size/2)])
    
    MSU_25_65keV = np.mean(Array_65keV[int(ROI_array[0,6]-square_size/2):int(ROI_array[0,6]+square_size/2), \
                                        int(ROI_array[1,6]-square_size/2):int(ROI_array[1,6]+square_size/2), \
                                        int(ROI_array[2,6]-square_size/2):int(ROI_array[2,6]+square_size/2)])
    MSU_25_50keV = np.mean(Array_50keV[int(ROI_array[0,6]-square_size/2):int(ROI_array[0,6]+square_size/2), \
                                        int(ROI_array[1,6]-square_size/2):int(ROI_array[1,6]+square_size/2), \
                                        int(ROI_array[2,6]-square_size/2):int(ROI_array[2,6]+square_size/2)])
    
    MSU_50_65keV = np.mean(Array_65keV[int(ROI_array[0,7]-square_size/2):int(ROI_array[0,7]+square_size/2), \
                                        int(ROI_array[1,7]-square_size/2):int(ROI_array[1,7]+square_size/2), \
                                        int(ROI_array[2,7]-square_size/2):int(ROI_array[2,7]+square_size/2)])
    MSU_50_50keV = np.mean(Array_50keV[int(ROI_array[0,7]-square_size/2):int(ROI_array[0,7]+square_size/2), \
                                        int(ROI_array[1,7]-square_size/2):int(ROI_array[1,7]+square_size/2), \
                                        int(ROI_array[2,7]-square_size/2):int(ROI_array[2,7]+square_size/2)])
    
    HA_100_65keV = np.mean(Array_65keV[int(ROI_array[0,0]-square_size/2):int(ROI_array[0,0]+square_size/2), \
                                        int(ROI_array[1,0]-square_size/2):int(ROI_array[1,0]+square_size/2), \
                                        int(ROI_array[2,0]-square_size/2):int(ROI_array[2,0]+square_size/2)])
    HA_100_50keV = np.mean(Array_50keV[int(ROI_array[0,0]-square_size/2):int(ROI_array[0,0]+square_size/2), \
                                        int(ROI_array[1,0]-square_size/2):int(ROI_array[1,0]+square_size/2), \
                                        int(ROI_array[2,0]-square_size/2):int(ROI_array[2,0]+square_size/2)])
    
    HA_400_65keV = np.mean(Array_65keV[int(ROI_array[0,1]-square_size/2):int(ROI_array[0,1]+square_size/2), \
                                        int(ROI_array[1,1]-square_size/2):int(ROI_array[1,1]+square_size/2), \
                                        int(ROI_array[2,1]-square_size/2):int(ROI_array[2,1]+square_size/2)])
    HA_400_50keV = np.mean(Array_50keV[int(ROI_array[0,1]-square_size/2):int(ROI_array[0,1]+square_size/2), \
                                        int(ROI_array[1,1]-square_size/2):int(ROI_array[1,1]+square_size/2), \
                                        int(ROI_array[2,1]-square_size/2):int(ROI_array[2,1]+square_size/2)])
    
    HA_800_65keV = np.mean(Array_65keV[int(ROI_array[0,2]-square_size/2):int(ROI_array[0,2]+square_size/2), \
                                        int(ROI_array[1,2]-square_size/2):int(ROI_array[1,2]+square_size/2), \
                                        int(ROI_array[2,2]-square_size/2):int(ROI_array[2,2]+square_size/2)])
    HA_800_50keV = np.mean(Array_50keV[int(ROI_array[0,2]-square_size/2):int(ROI_array[0,2]+square_size/2), \
                                        int(ROI_array[1,2]-square_size/2):int(ROI_array[1,2]+square_size/2), \
                                        int(ROI_array[2,2]-square_size/2):int(ROI_array[2,2]+square_size/2)])
    
    #------------------------------------------------------
    # Applying the Threshold 
    #------------------------------------------------------
    Array_Dual = Array_65keV
    #Array_Three = Array_65keV
    
    #Array_50keV = np.reshape(image_array_50keV,[xsize, ysize, zsize],order='F')
    #Array_50keV = np.flip(Array_50keV, axis=1)
    #Array_50keV = np.reshape(Array_50keV,[xsize* ysize* zsize],order='F')
    
    #Horizontal
    horizontal_50keV = MSU_25_50keV
    horizontal_65keV = MSU_25_65keV
    
    m_bone = (HA_800_65keV-HA_100_65keV)/(HA_800_50keV-HA_100_50keV) #The slope
    m_uric = (MSU_50_65keV-MSU_0_65keV)/(MSU_50_50keV-MSU_0_50keV)
    m = (m_bone+m_uric)/2
    b_bone = HA_100_65keV-m*HA_100_50keV
    y_point = MSU_25_65keV
    xbone_line = (horizontal_65keV-b_bone)/m
    x_point = (horizontal_50keV+xbone_line)/2
    #x_point = MSU_25_50keV
    b_line = y_point-m*x_point
    
    Array_Dual = 1*(image_array_65keV>image_array_50keV*m+b_line) + 1*(image_array_65keV>horizontal_50keV) >1
    Array_Dual = np.reshape(Array_Dual,[xsize, ysize, zsize], order='F')

    #------------------------------------------------------
    # Applying the Three Component Analysis 
    #------------------------------------------------------
    #Array_Dual_ST = Array_Dual*0
    #Array_Dual_UA = Array_Dual*0
    #Array_Dual_HA = Array_Dual*0
    ST_50keV = MSU_0_50keV
    ST_65keV = MSU_0_65keV
    
    UA_50keV = MSU_50_50keV
    UA_65keV = MSU_50_65keV
    
    HA_50keV = HA_800_50keV
    HA_65keV = HA_800_65keV
    
    if Correct_bone:
        #bone_percent=project_precent_a(HA_800_50keV,HA_800_65keV,2550,1700,1,1)
        
        perp_m = -2530/1710        
        perp_b = HA_800_65keV-perp_m*HA_800_50keV
        bone_m = 1710/2530
        HA_800_50keV = perp_b/(bone_m-perp_m)
        HA_800_65keV = HA_50keV*bone_m
        
        #HA_50keV = bone_percent*2550
        #HA_65keV = bone_percent*1700
    
    Z=np.zeros((xsize, ysize, zsize,3))
    #print('Applying 3 component decomp')
    if Extend_seed:
        UA_50keV,UA_65keV = extend_x_y(MSU_0_50keV, MSU_0_65keV, MSU_50_50keV, MSU_50_65keV, extension=1) 
        HA_50keV,HA_65keV = extend_x_y(HA_100_50keV, HA_100_65keV, HA_800_50keV, HA_800_65keV, extension=0.5)  
        #UA_50keV,UA_65keV = extend_x_y(MSU_0_50keV,MSU_0_65keV,MSU_50_50keV,MSU_50_65keV,extension=0.01) 
        #HA_50keV,HA_65keV = extend_x_y(HA_100_50keV,HA_100_65keV,HA_800_50keV,HA_800_65keV,extension=0.01)            
        print(UA_50keV)
        print(UA_65keV)
        print(HA_50keV)
        print(HA_65keV)
    #print('Solving the linear equations')
    ST_50keV = float(ST_50keV)
    ST_65keV = float(ST_65keV)
    
    UA_50keV = float(UA_50keV)
    UA_65keV = float(UA_65keV)
    
    HA_50keV = float(HA_50keV)
    HA_65keV = float(HA_65keV)
    #Doing it without for loops to speed things up by around 40 minutes
    a = ST_50keV
    b = UA_50keV
    c = HA_50keV
    d = Array_50keV
    e = ST_65keV
    f = UA_65keV
    g = HA_65keV
    h = Array_65keV              
    Z[:,:,:,1] = ((d-a)-(h-e)*(c-a)/(g-e))/((b-a)-(c-a)*(f-e)/(g-e)) 
    Z[:,:,:,2] = ((d-a)-(h-e)*(b-a)/(f-e))/((c-a)-(b-a)*(g-e)/(f-e))
    Z[:,:,:,0] = 1-Z[:,:,:,2]-Z[:,:,:,1]
    #Z_hold1
    Z = Z.astype(float)
    Array_50keV = Array_50keV.astype(float)
    # Removing Air
    
    print(np.average(Z[:,:,:,0]+Z[:,:,:,1]+Z[:,:,:,2]))
    
    Z[Array_50keV<-300.00,:] = 0
    #print('Removing the double negatives')
    #Everything with a double negative is kept
    #Z_DN=(Z[:,:,:,0]*Z[:,:,:,1]*Z[:,:,:,2])<0 #This was wrong!
    Z_DN = ((Z[:,:,:,0]*Z[:,:,:,1]*Z[:,:,:,2])>0) & ((np.absolute(Z[:,:,:,0])+np.absolute(Z[:,:,:,1])+np.absolute(Z[:,:,:,2]))>1)
    Z_S = ((Z[:,:,:,0]*Z[:,:,:,1]*Z[:,:,:,2])>0) & ((np.absolute(Z[:,:,:,0])+np.absolute(Z[:,:,:,1])+np.absolute(Z[:,:,:,2]))==1)

    print(np.sum(Z_DN))
    #Everything Negative is set to zero
    #print(np.average(Z[:,:,0]))
    #print(np.average(Z[:,:,:,1]))
    #print(np.average(Z[:,:,2]))
    
    
    #And set the remaining positive to 1 if it had a double negative
    Z[(Z_DN) & (Z[:,:,:,0]>1.00),0] = 1.00
    Z[(Z_DN) & (Z[:,:,:,1]>1.00),1] = 1.00
    Z[(Z_DN) & (Z[:,:,:,2]>1.00),2] = 1.00
    
    #print(np.average(Z[:,:,0]))
    #print(np.average(Z[:,:,:,1]))
    #print(np.average(Z[:,:,2]))
    Z[Z<0] = 0
    #print('Finding points to be interpolated')
    #Itentify the points that need to be interpolated, the ones with two numbers!
    Z_TS = ((Z[:,:,:,1]+Z[:,:,:,2])>1.00) & (Z[:,:,:,1]*Z[:,:,:,2]>0)
    Z_UA = ((Z[:,:,:,0]+Z[:,:,:,2])>1.00) & (Z[:,:,:,0]*Z[:,:,:,2]>0)
    Z_HA = ((Z[:,:,:,0]+Z[:,:,:,1])>1.00) & (Z[:,:,:,0]*Z[:,:,:,1]>0)
    
    print(np.sum(Z_TS * Z_DN))
    print(np.sum(Z_UA * Z_DN))
    print(np.sum(Z_HA * Z_DN))
    print(np.sum(Z_HA * Z_UA))
    print(np.sum(Z_HA * Z_TS))
    print(np.sum(Z_UA * Z_TS))
    print(np.average(Z_DN | Z_HA |  Z_UA |  Z_TS | Z_S | (Array_50keV<-300.00)))
    print(np.max(Z[:,:,0]))
    print(np.max(Z[:,:,:,1]))
    print(np.max(Z[:,:,2]))
    print(np.min(Z[:,:,0]))
    print(np.min(Z[:,:,:,1]))
    print(np.min(Z[:,:,2]))
    #print('Finding the zero TSs')
    Z[Z_TS,1] = project_precent_a_array(Array_50keV[Z_TS],Array_65keV[Z_TS],UA_50keV,UA_65keV,HA_50keV,HA_65keV)
    Z[Z_TS,2] = 1-Z[Z_TS,1]
    
    
    #print('Finding the zero UAs')
    Z[Z_UA,0] = project_precent_a_array(Array_50keV[Z_UA],Array_65keV[Z_UA],ST_50keV,ST_65keV,HA_50keV,HA_65keV)
    Z[Z_UA,2] = 1-Z[Z_UA,0]
    
    
    #print('Finding the zero HAs')
    Z[Z_HA,0] = project_precent_a_array(Array_50keV[Z_HA],Array_65keV[Z_HA],ST_50keV,ST_65keV,UA_50keV,UA_65keV)
    Z[Z_HA,1] = 1-Z[Z_HA,0]
    
    print(np.max(Z[:,:,0]))
    print(np.max(Z[:,:,:,1]))
    print(np.max(Z[:,:,2]))
    print(np.min(Z[:,:,0]))
    print(np.min(Z[:,:,:,1]))
    print(np.min(Z[:,:,2]))
    z_image = int(ROI_array[2,7]+0.5)
    #Save image with the same z slice as the 50% UA, I need to rotate the image to get the same as the original
    im = Image.fromarray(np.rot90((Z[:,:,z_image,0]*255).astype(np.uint8)))
    im.save('/Users/mkuczyns/Projects/Gout/output/seg/Patient'+ Patient+'Leg_'+Leg+'_ST.png')
    im = Image.fromarray(np.rot90((Z[:,:,z_image,1]*255).astype(np.uint8)))
    im.save('/Users/mkuczyns/Projects/Gout/output/seg/Patient'+ Patient+'Leg_'+Leg+'_UA.png')   
    im = Image.fromarray(np.rot90((Z[:,:,z_image,2]*255).astype(np.uint8)))
    im.save('/Users/mkuczyns/Projects/Gout/output/seg/Patient'+ Patient+'Leg_'+Leg+'_HA.png')
    im = Image.fromarray(np.rot90((((Array_50keV[:,:,z_image]-np.min(Array_50keV))/(np.max(Array_50keV)-np.min(Array_50keV)))*255).astype(np.uint8)))
    im.save('/Users/mkuczyns/Projects/Gout/output/seg/Patient'+ Patient+'Leg_'+Leg+'_50keV.png')
    im = Image.fromarray(np.rot90((((Array_65keV[:,:,z_image]-np.min(Array_65keV))/(np.max(Array_65keV)-np.min(Array_65keV)))*255).astype(np.uint8)))
    im.save('/Users/mkuczyns/Projects/Gout/output/seg/Patient'+ Patient+'Leg_'+Leg+'_65keV.png')
    im = Image.fromarray(np.rot90((Array_Dual[:,:,z_image]*255).astype(np.uint8)))
    im.save('/Users/mkuczyns/Projects/Gout/output/seg/Patient'+ Patient+'Leg_'+Leg+'_Dual.png')
    # im = Image.fromarray(np.rot90((Z[:,:,z_image,0]*255).astype(np.uint8)))
    # im.save('C:/Users/dakondro.UC/Documents/Gout-Segmented_Images/Patient'+ Patient+'Leg_'+Leg+'_ST.png')
    # im = Image.fromarray(np.rot90((Z[:,:,z_image,1]*255).astype(np.uint8)))
    # im.save('C:/Users/dakondro.UC/Documents/Gout-Segmented_Images/Patient'+ Patient+'Leg_'+Leg+'_UA.png')   
    # im = Image.fromarray(np.rot90((Z[:,:,z_image,2]*255).astype(np.uint8)))
    # im.save('C:/Users/dakondro.UC/Documents/Gout-Segmented_Images/Patient'+ Patient+'Leg_'+Leg+'_HA.png')
    # im = Image.fromarray(np.rot90((((Array_50keV[:,:,z_image]-np.min(Array_50keV))/(np.max(Array_50keV)-np.min(Array_50keV)))*255).astype(np.uint8)))
    # im.save('C:/Users/dakondro.UC/Documents/Gout-Segmented_Images/Patient'+ Patient+'Leg_'+Leg+'_50keV.png')
    # im = Image.fromarray(np.rot90((((Array_65keV[:,:,z_image]-np.min(Array_65keV))/(np.max(Array_65keV)-np.min(Array_65keV)))*255).astype(np.uint8)))
    # im.save('C:/Users/dakondro.UC/Documents/Gout-Segmented_Images/Patient'+ Patient+'Leg_'+Leg+'_65keV.png')
    # im = Image.fromarray(np.rot90((Array_Dual[:,:,z_image]*255).astype(np.uint8)))
    # im.save('C:/Users/dakondro.UC/Documents/Gout-Segmented_Images/Patient'+ Patient+'Leg_'+Leg+'_Dual.png')
    
    
    UA_Array = Z[:,:,:,1]*1000
    
    #Convert to meaningful values
    
    if Extend_seed:
        Z[:,:,:,1] = 652*2*Z[:,:,:,1]
        Z[:,:,:,2] = 800*(1+(0.5*7/8))*Z[:,:,:,2]  
        #Z[:,:,:,1]=652*1.01*Z[:,:,:,1]
        #Z[:,:,:,2]=800*(1.01)*Z[:,:,:,2] 
    else:
        Z[:,:,:,1] = 652*Z[:,:,:,1]
        Z[:,:,:,2] = 800*Z[:,:,:,2]
    
    #---------------------------------------------------
    # Saving Stuff 
    #---------------------------------------------------
    
    #Finding the values of everything
    MSU_0_Seg = np.mean(Z[int(ROI_array[0,3]-square_size/2):int(ROI_array[0,3]+square_size/2),int(ROI_array[1,3]-square_size/2):int(ROI_array[1,3]+square_size/2),int(ROI_array[2,3]-square_size/2):int(ROI_array[2,3]+square_size/2),1])
    MSU_0625_Seg = np.mean(Z[int(ROI_array[0,4]-square_size/2):int(ROI_array[0,4]+square_size/2),int(ROI_array[1,4]-square_size/2):int(ROI_array[1,4]+square_size/2),int(ROI_array[2,4]-square_size/2):int(ROI_array[2,4]+square_size/2),1])
    MSU_125_Seg = np.mean(Z[int(ROI_array[0,5]-square_size/2):int(ROI_array[0,5]+square_size/2),int(ROI_array[1,5]-square_size/2):int(ROI_array[1,5]+square_size/2),int(ROI_array[2,5]-square_size/2):int(ROI_array[2,5]+square_size/2),1])
    MSU_25_Seg = np.mean(Z[int(ROI_array[0,6]-square_size/2):int(ROI_array[0,6]+square_size/2),int(ROI_array[1,6]-square_size/2):int(ROI_array[1,6]+square_size/2),int(ROI_array[2,6]-square_size/2):int(ROI_array[2,6]+square_size/2),1])
    MSU_50_Seg = np.mean(Z[int(ROI_array[0,7]-square_size/2):int(ROI_array[0,7]+square_size/2),int(ROI_array[1,7]-square_size/2):int(ROI_array[1,7]+square_size/2),int(ROI_array[2,7]-square_size/2):int(ROI_array[2,7]+square_size/2),1])
    HA_100_Seg = np.mean(Z[int(ROI_array[0,0]-square_size/2):int(ROI_array[0,0]+square_size/2),int(ROI_array[1,0]-square_size/2):int(ROI_array[1,0]+square_size/2),int(ROI_array[2,0]-square_size/2):int(ROI_array[2,0]+square_size/2),2])
    HA_400_Seg = np.mean(Z[int(ROI_array[0,1]-square_size/2):int(ROI_array[0,1]+square_size/2),int(ROI_array[1,1]-square_size/2):int(ROI_array[1,1]+square_size/2),int(ROI_array[2,1]-square_size/2):int(ROI_array[2,1]+square_size/2),2])
    HA_800_Seg = np.mean(Z[int(ROI_array[0,2]-square_size/2):int(ROI_array[0,2]+square_size/2),int(ROI_array[1,2]-square_size/2):int(ROI_array[1,2]+square_size/2),int(ROI_array[2,2]-square_size/2):int(ROI_array[2,2]+square_size/2),2])
    
    MSU_0_Seg_dual = np.mean(Array_Dual[int(ROI_array[0,3]-square_size/2):int(ROI_array[0,3]+square_size/2),int(ROI_array[1,3]-square_size/2):int(ROI_array[1,3]+square_size/2),int(ROI_array[2,3]-square_size/2):int(ROI_array[2,3]+square_size/2)])
    MSU_0625_Seg_dual = np.mean(Array_Dual[int(ROI_array[0,4]-square_size/2):int(ROI_array[0,4]+square_size/2),int(ROI_array[1,4]-square_size/2):int(ROI_array[1,4]+square_size/2),int(ROI_array[2,4]-square_size/2):int(ROI_array[2,4]+square_size/2)])
    MSU_125_Seg_dual = np.mean(Array_Dual[int(ROI_array[0,5]-square_size/2):int(ROI_array[0,5]+square_size/2),int(ROI_array[1,5]-square_size/2):int(ROI_array[1,5]+square_size/2),int(ROI_array[2,5]-square_size/2):int(ROI_array[2,5]+square_size/2)])
    MSU_25_Seg_dual = np.mean(Array_Dual[int(ROI_array[0,6]-square_size/2):int(ROI_array[0,6]+square_size/2),int(ROI_array[1,6]-square_size/2):int(ROI_array[1,6]+square_size/2),int(ROI_array[2,6]-square_size/2):int(ROI_array[2,6]+square_size/2)])
    MSU_50_Seg_dual = np.mean(Array_Dual[int(ROI_array[0,7]-square_size/2):int(ROI_array[0,7]+square_size/2),int(ROI_array[1,7]-square_size/2):int(ROI_array[1,7]+square_size/2),int(ROI_array[2,7]-square_size/2):int(ROI_array[2,7]+square_size/2)])
    #print(MSU_0_Seg)
    #print(MSU_0625_Seg)
    #print(MSU_125_Seg)
    #print(MSU_25_Seg)
    #print(MSU_50_Seg)
    #print(HA_100_Seg)
    #print(HA_400_Seg)
    #print(HA_800_Seg)
    #Array2Copy = vtk.vtkImageData()
    #Array2Copy.DeepCopy(medicalImage)  
    
    #vtk_Array_UA=np.reshape(UA_Array,[xsize* ysize* zsize],order='F')
    
    #vtk_data_array = numpy_to_vtk(vtk_Array_UA)
    
    #image = vtk.vtkImageData()
    
    #points = image.GetPointData()
    #points.SetScalars(vtk_data_array)
    
    #image = vtk.vtkImageData()
    #image.SetDimensions((xsize,ysize,zsize))
    
    output = np.array([[Patient,Leg,MSU_0_Seg_dual,MSU_0625_Seg_dual,MSU_125_Seg_dual,MSU_25_Seg_dual,MSU_50_Seg_dual,\
                      MSU_0_Seg,MSU_0625_Seg,MSU_125_Seg,MSU_25_Seg,MSU_50_Seg,\
                      HA_100_Seg,HA_400_Seg,HA_800_Seg,\
                      MSU_0_50keV, MSU_0625_50keV, MSU_125_50keV,MSU_25_50keV,MSU_50_50keV,\
                      HA_100_50keV, HA_400_50keV, HA_800_50keV,\
                      MSU_0_65keV, MSU_0625_65keV, MSU_125_65keV,MSU_25_65keV,MSU_50_65keV,\
                      HA_100_65keV, HA_400_65keV, HA_800_65keV]], dtype=object)
    #print(np.size(output))
    #if (write):
    #    print('Writing segementation to output file...')
     
    #   writer = vtk.vtkNIFTIImageWriter()
    #   writer.SetInputData(image)
    #    writer.SetFileName(os.path.join(outpath, 'Written.nii'))
        # copy most information directory from the header
        #writer.SetNIFTIHeader(reader50keV.GetNIFTIHeader())
        # this information will override the reader's header
        #writer.SetQFac(reader50keV.GetQFac())
        #writer.SetTimeDimension(reader50keV.GetTimeDimension())
        #writer.SetQFormMatrix(reader50keV.GetQFormMatrix())
        #writer.SetSFormMatrix(reader50keV.GetSFormMatrix())
    #    writer.Write()
    
    return output