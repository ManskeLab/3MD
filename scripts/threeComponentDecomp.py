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
import sys
import numpy as np

from PIL import Image

import SimpleITK as sitk

import vtk
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from projectPercent import project_precent_a_array, extend_x_y

def calculateMean(array, ROI_array, square_size, a, b, c, d, e, f):
    mean = np.mean(array[int(ROI_array[a,b] - square_size/2):int(ROI_array[a,b] + square_size/2), \
                            int(ROI_array[c,d] - square_size/2):int(ROI_array[c,d] + square_size/2), \
                            int(ROI_array[e,f] - square_size/2):int(ROI_array[e,f] + square_size/2)])
    return mean

def sitk2vtk(sitkImage, orientation, spacing, sliceNumber):
    # Convert SimpleITK image to numpy array
    npArray = sitk.GetArrayFromImage(sitkImage)
    
    # Get the image dimensions
    dz, dy, dx = npArray.shape

    # Convert from numpy to VTK
    vtkImage = numpy_support.numpy_to_vtk(npArray.flat)

    # Setup the image properties
    # Image orientation should be AXIAL since this is CT...
    if orientation == 'AXIAL':
        extent = (0, dx -1, 0, dy -1, sliceNumber, sliceNumber + dz - 1)
    elif orientation == 'SAGITAL':
        extent = (sliceNumber, sliceNumber + dx - 1, 0, dy - 1, 0, dz - 1)
    elif orientation == 'CORONAL':
        extent = (0, dx - 1, sliceNumber, sliceNumber + dy - 1, 0, dz - 1)

    # Create the VTK image object
    image = vtk.vtkImageData()
    image.SetOrigin(0, 0, 0)
    image.SetSpacing(spacing)
    image.SetDimensions(dx, dy, dz)
    image.SetExtent(extent)
    image.AllocateScalars(numpy_support.get_vtk_array_type(npArray.dtype), 1)
    image.GetCellData().SetScalars(vtkImage)
    image.GetPointData().SetScalars(vtkImage)

    vtkImageCopy = vtk.vtkImageData()
    vtkImageCopy.DeepCopy(image)

    return vtkImageCopy


def Component3_Decomp(dicomPath_50keV, dicomPath_65keV, Patient, Side, filterimage=True, Correct_bone=True, Extend_seed=True):                
    parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outpath = os.path.join(parentDir, 'output')

    # Load the two images I expect that they are dicoms
    reader50keV = sitk.ImageSeriesReader()
    dicom_names = reader50keV.GetGDCMSeriesFileNames(dicomPath_50keV)
    reader50keV.SetFileNames(dicom_names)
    image50keV = reader50keV.Execute()

    reader65keV = sitk.ImageSeriesReader()
    dicom_names = reader65keV.GetGDCMSeriesFileNames(dicomPath_65keV)
    reader65keV.SetFileNames(dicom_names)
    image65keV = reader65keV.Execute()

    # Enforce matching dimensions
    (dx, dy, dz) = image50keV.GetSize()
    (xsize, ysize, zsize) = image65keV.GetSize()
    
    if dx != xsize and dy != ysize and dz != zsize:
        print('Error: Dimensions of both images must match!')
        sys.exit(1)

    vtkImage50keV = sitk2vtk(image50keV, 'AXIAL', image50keV.GetSpacing(), 0)
    vtkImage65keV = sitk2vtk(image65keV, 'AXIAL', image65keV.GetSpacing(), 0)

    # print('Should we filter?')
    if filterimage:
        # print('Applying Gaussian Smooth') 
        filter_image = vtk.vtkImageGaussianSmooth()
        filter_image.SetInputData(vtkImage50keV)
        filter_image.SetStandardDeviation(2)
        filter_image.SetRadiusFactors(3,3,3)
        filter_image.SetDimensionality(3)
        filter_image.Update()
        nodes_vtk_array_50 = filter_image.GetOutput().GetPointData().GetArray(0)

        filter_image.SetInputData(vtkImage65keV)
        filter_image.Update()
        nodes_vtk_array_65 = filter_image.GetOutput().GetPointData().GetArray(0)
    else:
        # print('No Filter Applied')
        nodes_vtk_array_50 = vtkImage50keV.GetPointData().GetScalars()
        nodes_vtk_array_65 = vtkImage65keV.GetPointData().GetScalars()

    image_array = np.zeros((int(dx * dy * dz), 4))    
    image_array_50keV = vtk_to_numpy(nodes_vtk_array_50)
    image_array_65keV = vtk_to_numpy(nodes_vtk_array_65)
    
    Array_65keV = np.reshape(image_array_65keV, [dx, dy, dz], order='F')
    Array_50keV = np.reshape(image_array_50keV, [dx, dy, dz], order='F')
    Array_Dual = Array_65keV
    Array_Three = Array_65keV
    
    
    #-------------------------------------------------------
    # ROI Extration
    #--------------------------------------------------------
    square_size = 11
    ROI_list = []
    
    # Get the CSV input file. Assume the file name is PatientN_Phantom_ROIs.csv, where N is the patient number.
    phantomROIsCSV = os.path.join(os.path.join(parentDir, 'input'), 'Phantom_ROIs.csv')

    with open(phantomROIsCSV, newline = '') as ROIs:                                                                                      
        ROI_reader = csv.reader(ROIs, delimiter=',')
        for ROI in ROI_reader:
            ROI_list.append(ROI)
    ROI_array = np.array(ROI_list)
    
    ROI_array = ROI_array[1:,:] # slice out the top row (i.e., headers)
    ROI_array = ROI_array[ROI_array[:,0]==Patient,:] # Compares values, elementwise, between 'Patient' and first column of 'ROI_array'. Remove data for patients not in current 'Patient'
    ROI_array = ROI_array[ROI_array[:,1]==Side,:] # Same as above but for right or left leg
    ROI_array = ROI_array[:,3:-1] # Slice out first two columns and last column. Not sure what is in these columns...
    ROI_array = ROI_array.astype(float)

    print('*** Printing ROI array...')
    print('ROI_array: ' + str(ROI_array))
    print()
    
    MSU_0_65keV     = calculateMean(Array_65keV, ROI_array, square_size, 0, 3, 1, 3, 2, 3)
    MSU_0_50keV     = calculateMean(Array_50keV, ROI_array, square_size, 0, 3, 1, 3, 2, 3)
    MSU_0625_65keV  = calculateMean(Array_65keV, ROI_array, square_size, 0, 4, 1, 4, 2, 4)
    MSU_0625_50keV  = calculateMean(Array_50keV, ROI_array, square_size, 0, 4, 1, 4, 2, 4)
    MSU_125_65keV   = calculateMean(Array_65keV, ROI_array, square_size, 0, 5, 1, 5, 2, 5)
    MSU_125_50keV   = calculateMean(Array_50keV, ROI_array, square_size, 0, 5, 1, 5, 2, 5)
    MSU_25_65keV    = calculateMean(Array_65keV, ROI_array, square_size, 0, 6, 1, 6, 2, 6)
    MSU_25_50keV    = calculateMean(Array_50keV, ROI_array, square_size, 0, 6, 1, 6, 2, 6)
    MSU_50_65keV    = calculateMean(Array_65keV, ROI_array, square_size, 0, 7, 1, 7, 2, 7)
    MSU_50_50keV    = calculateMean(Array_50keV, ROI_array, square_size, 0, 7, 1, 7, 2, 7)
    HA_100_65keV    = calculateMean(Array_65keV, ROI_array, square_size, 0, 0, 1, 0, 2, 0)
    HA_100_50keV    = calculateMean(Array_50keV, ROI_array, square_size, 0, 0, 1, 0, 2, 0)
    HA_400_65keV    = calculateMean(Array_65keV, ROI_array, square_size, 0, 1, 1, 1, 2, 1)
    HA_400_50keV    = calculateMean(Array_50keV, ROI_array, square_size, 0, 1, 1, 1, 2, 1)
    HA_800_65keV    = calculateMean(Array_65keV, ROI_array, square_size, 0, 2, 1, 2, 2, 2)
    HA_800_50keV    = calculateMean(Array_50keV, ROI_array, square_size, 0, 2, 1, 2, 2, 2)
    
    print('*** Printing mean of ROIs...')
    print('MSU_0_65keV: ' + str(MSU_0_65keV))
    print('MSU_0_50keV: ' + str(MSU_0_50keV))
    print('MSU_0625_65keV: ' + str(MSU_0625_65keV))
    print('MSU_0625_50keV: ' + str(MSU_0625_50keV))
    print('MSU_125_65keV: ' + str(MSU_125_65keV))
    print('MSU_125_50keV: ' + str(MSU_125_50keV))
    print('MSU_25_65keV: ' + str(MSU_25_65keV))
    print('MSU_25_50keV: ' + str(MSU_25_50keV))
    print('MSU_50_65keV: ' + str(MSU_50_65keV))
    print('MSU_50_50keV: ' + str(MSU_50_50keV))
    print('HA_100_65keV: ' + str(HA_100_65keV))
    print('HA_100_50keV: ' + str(HA_100_50keV))
    print('HA_400_65keV: ' + str(HA_400_65keV))
    print('HA_400_50keV: ' + str(HA_400_50keV))
    print('HA_800_65keV: ' + str(HA_800_65keV))
    print('HA_800_50keV: ' + str(HA_800_50keV))
    print()


    #------------------------------------------------------
    # Applying the Threshold 
    #------------------------------------------------------
    Array_Dual = Array_65keV
    # Array_Three = Array_65keV
    
    #Array_50keV = np.reshape(image_array_50keV,[dx, dy, dz],order='F')
    #Array_50keV = np.flip(Array_50keV, axis=1)
    #Array_50keV = np.reshape(Array_50keV,[dx* dy* dz],order='F')
    
    #Horizontal
    horizontal_50keV = MSU_25_50keV
    horizontal_65keV = MSU_25_65keV
    
    m_bone = (HA_800_65keV - HA_100_65keV) / (HA_800_50keV - HA_100_50keV) # The slope
    m_uric = (MSU_50_65keV - MSU_0_65keV) / (MSU_50_50keV - MSU_0_50keV)
    m = (m_bone + m_uric) / 2
    b_bone = HA_100_65keV - m * HA_100_50keV
    y_point = MSU_25_65keV
    xbone_line = (horizontal_65keV - b_bone) / m
    x_point = (horizontal_50keV + xbone_line) / 2
    # x_point = MSU_25_50keV
    b_line = y_point-m * x_point
    
    Array_Dual = 1*(image_array_65keV > image_array_50keV * m + b_line) + 1*(image_array_65keV > horizontal_50keV) > 1
    Array_Dual = np.reshape(Array_Dual,[dx, dy, dz], order='F')

    #------------------------------------------------------
    # Applying the Three Component Analysis 
    #------------------------------------------------------
    # Array_Dual_ST = Array_Dual * 0
    # Array_Dual_UA = Array_Dual * 0
    # Array_Dual_HA = Array_Dual * 0
    ST_50keV = MSU_0_50keV
    ST_65keV = MSU_0_65keV
    
    UA_50keV = MSU_50_50keV
    UA_65keV = MSU_50_65keV
    
    HA_50keV = HA_800_50keV
    HA_65keV = HA_800_65keV
    
    if Correct_bone:
        # bone_percent = project_precent_a(HA_800_50keV, HA_800_65keV, 2550, 1700, 1, 1)
        perp_m = -2530 / 1710        
        perp_b = HA_800_65keV - perp_m * HA_800_50keV
        bone_m = 1710 / 2530
        HA_800_50keV = perp_b / (bone_m - perp_m)
        HA_800_65keV = HA_50keV * bone_m
        
        # HA_50keV = bone_percent * 2550
        # HA_65keV = bone_percent * 1700
    
    Z = np.zeros((dx, dy, dz,3))
    # print('Applying 3 component decomp')
    if Extend_seed:
        UA_50keV, UA_65keV = extend_x_y(MSU_0_50keV, MSU_0_65keV, MSU_50_50keV, MSU_50_65keV, extension=1) 
        HA_50keV, HA_65keV = extend_x_y(HA_100_50keV, HA_100_65keV, HA_800_50keV, HA_800_65keV, extension=0.5)  
        # UA_50keV, UA_65keV = extend_x_y(MSU_0_50keV, MSU_0_65keV, MSU_50_50keV, MSU_50_65keV, extension=0.01) 
        # HA_50keV, HA_65keV = extend_x_y(HA_100_50keV, HA_100_65keV, HA_800_50keV, HA_800_65keV, extension=0.01)            
        # print('UA_50keV = ' + str(UA_50keV))
        # print('UA_65keV = ' + str(UA_65keV))
        # print('HA_50keV = ' + str(HA_50keV))
        # print('HA_65keV = ' + str(HA_65keV))

    # print('Solving the linear equations')
    ST_50keV = float(ST_50keV)
    ST_65keV = float(ST_65keV)
    
    UA_50keV = float(UA_50keV)
    UA_65keV = float(UA_65keV)
    
    HA_50keV = float(HA_50keV)
    HA_65keV = float(HA_65keV)

    # Doing it without for loops to speed things up by around 40 minutes
    print('*** Performing 3MD...')
    a = ST_50keV
    b = UA_50keV
    c = HA_50keV
    d = Array_50keV
    e = ST_65keV
    f = UA_65keV
    g = HA_65keV
    h = Array_65keV              
    Z[:,:,:,1] = ( (d-a)-(h-e) * (c-a) / (g-e) ) / ( (b-a)-(c-a) * (f-e) / (g-e) ) 
    Z[:,:,:,2] = ( (d-a)-(h-e) * (b-a) / (f-e) ) / ( (c-a)-(b-a) * (g-e) / (f-e) )
    Z[:,:,:,0] = 1 - Z[:,:,:,2] - Z[:,:,:,1]
    # Z_hold1
    Z = Z.astype(float)
    Array_50keV = Array_50keV.astype(float)
    
    # Removing Air
    # print(np.average(Z[:,:,:,0] + Z[:,:,:,1] + Z[:,:,:,2]))
    
    Z[Array_50keV<-300.00,:] = 0
    # print('Removing the double negatives')
    # Everything with a double negative is kept
    # Z_DN = (Z[:,:,:,0] * Z[:,:,:,1] * Z[:,:,:,2]) < 0 # This was wrong!
    Z_DN = ((Z[:,:,:,0] * Z[:,:,:,1] * Z[:,:,:,2]) > 0) & ((np.absolute(Z[:,:,:,0]) + np.absolute(Z[:,:,:,1]) + np.absolute(Z[:,:,:,2])) > 1)
    Z_S = ((Z[:,:,:,0] * Z[:,:,:,1] * Z[:,:,:,2]) > 0) & ((np.absolute(Z[:,:,:,0]) + np.absolute(Z[:,:,:,1]) + np.absolute(Z[:,:,:,2])) == 1)

    # print(np.sum(Z_DN))
    # Everything Negative is set to zero
    # print(np.average(Z[:,:,0]))
    # print(np.average(Z[:,:,:,1]))
    # print(np.average(Z[:,:,2]))
    
    # And set the remaining positive to 1 if it had a double negative
    Z[(Z_DN) & (Z[:,:,:,0]>1.00),0] = 1.00
    Z[(Z_DN) & (Z[:,:,:,1]>1.00),1] = 1.00
    Z[(Z_DN) & (Z[:,:,:,2]>1.00),2] = 1.00
    
    # print(np.average(Z[:,:,0]))
    # print(np.average(Z[:,:,:,1]))
    # print(np.average(Z[:,:,2]))
    Z[Z<0] = 0
    # print('Finding points to be interpolated')
    # Itentify the points that need to be interpolated, the ones with two numbers!
    Z_TS = ((Z[:,:,:,1] + Z[:,:,:,2]) > 1.00) & (Z[:,:,:,1] * Z[:,:,:,2] > 0)
    Z_UA = ((Z[:,:,:,0] + Z[:,:,:,2]) > 1.00) & (Z[:,:,:,0] * Z[:,:,:,2] > 0)
    Z_HA = ((Z[:,:,:,0] + Z[:,:,:,1]) > 1.00) & (Z[:,:,:,0] * Z[:,:,:,1] > 0)
    
    # print(np.sum(Z_TS * Z_DN))
    # print(np.sum(Z_UA * Z_DN))
    # print(np.sum(Z_HA * Z_DN))
    # print(np.sum(Z_HA * Z_UA))
    # print(np.sum(Z_HA * Z_TS))
    # print(np.sum(Z_UA * Z_TS))
    # print(np.average(Z_DN | Z_HA |  Z_UA |  Z_TS | Z_S | (Array_50keV<-300.00)))
    # print(np.max(Z[:,:,0]))
    # print(np.max(Z[:,:,:,1]))
    # print(np.max(Z[:,:,2]))
    # print(np.min(Z[:,:,0]))
    # print(np.min(Z[:,:,:,1]))
    # print(np.min(Z[:,:,2]))

    # print('Finding the zero TSs')
    Z[Z_TS,1] = project_precent_a_array(Array_50keV[Z_TS], Array_65keV[Z_TS], UA_50keV, UA_65keV, HA_50keV, HA_65keV)
    Z[Z_TS,2] = 1 - Z[Z_TS,1]
    
    # print('Finding the zero UAs')
    Z[Z_UA,0] = project_precent_a_array(Array_50keV[Z_UA], Array_65keV[Z_UA], ST_50keV, ST_65keV, HA_50keV, HA_65keV)
    Z[Z_UA,2] = 1 - Z[Z_UA,0]
    
    # print('Finding the zero HAs')
    Z[Z_HA,0] = project_precent_a_array(Array_50keV[Z_HA], Array_65keV[Z_HA], ST_50keV, ST_65keV, UA_50keV, UA_65keV)
    Z[Z_HA,1] = 1 - Z[Z_HA,0]
    
    # print(np.max(Z[:,:,0]))
    # print(np.max(Z[:,:,:,1]))
    # print(np.max(Z[:,:,2]))
    # print(np.min(Z[:,:,0]))
    # print(np.min(Z[:,:,:,1]))
    # print(np.min(Z[:,:,2]))
    z_image = int(ROI_array[2,7] + 0.5)

    print('*** Done!')
    print()

    # Save image with the same z slice as the 50% UA, I need to rotate the image to get the same as the original
    # Create a new directory for each patient, if needed
    newPatientPath = os.path.join(outpath, 'Patient' + Patient)
    if not os.path.exists(newPatientPath):
        os.makedirs(newPatientPath)
    
    print('*** Saving output images...')
    im = Image.fromarray(np.rot90((Z[:,:,z_image,0] * 255).astype(np.uint8)))
    im.save(os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_ST.png'))
    im = Image.fromarray(np.rot90((Z[:,:,z_image,1] * 255).astype(np.uint8)))
    im.save(os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_UA.png'))  
    im = Image.fromarray(np.rot90((Z[:,:,z_image,2] * 255).astype(np.uint8)))
    im.save(os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_HA.png'))
    im = Image.fromarray(np.rot90((((Array_50keV[:,:,z_image] - np.min(Array_50keV)) / (np.max(Array_50keV) - np.min(Array_50keV))) * 255).astype(np.uint8)))
    im.save(os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_50keV.png'))
    im = Image.fromarray(np.rot90((((Array_65keV[:,:,z_image] - np.min(Array_65keV)) / (np.max(Array_65keV) - np.min(Array_65keV))) * 255).astype(np.uint8)))
    im.save(os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_65keV.png'))
    im = Image.fromarray(np.rot90((Array_Dual[:,:,z_image] * 255).astype(np.uint8)))
    im.save(os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_Dual.png'))

    # Save the 3D images
    # We should have 7 images total
    im = sitk.GetImageFromArray(np.rot90((Z[:,:,:,0] * 255).astype(np.uint8)))
    sitk.WriteImage(im, os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_ST.nii'))
    im = sitk.GetImageFromArray(np.rot90((Z[:,:,:,1] * 255).astype(np.uint8)))
    sitk.WriteImage(im, os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_UA.nii'))  
    im = sitk.GetImageFromArray(np.rot90((Z[:,:,:,2] * 255).astype(np.uint8)))
    sitk.WriteImage(im, os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_HA.nii'))
    im = sitk.GetImageFromArray(np.rot90((((Array_50keV[:,:,:] - np.min(Array_50keV)) / (np.max(Array_50keV) - np.min(Array_50keV))) * 255).astype(np.uint8)))
    sitk.WriteImage(im, os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_50keV.nii'))
    im = sitk.GetImageFromArray(np.rot90((((Array_65keV[:,:,:] - np.min(Array_65keV)) / (np.max(Array_65keV) - np.min(Array_65keV))) * 255).astype(np.uint8)))
    sitk.WriteImage(im, os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_65keV.nii'))
    im = sitk.GetImageFromArray(np.rot90((Array_Dual[:,:,:] * 255).astype(np.uint8)))
    sitk.WriteImage(im, os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_Dual.nii'))
    im2 = sitk.InvertIntensity(im)
    sitk.WriteImage(im2, os.path.join(newPatientPath, 'Patient' + Patient + 'Side_' + Side + '_Dual_Sub.nii'))

    print('*** Done!')
    print()
    
    # UA_Array = Z[:,:,:,1] * 1000
    
    # Convert to meaningful values
    # if Extend_seed:
    #     Z[:,:,:,1] = 652 * 2 * Z[:,:,:,1]
    #     Z[:,:,:,2] = 800 * (1 + (0.5 * 7/8)) * Z[:,:,:,2]  
    #     # Z[:,:,:,1] = 652 * 1.01 * Z[:,:,:,1]
    #     # Z[:,:,:,2] = 800 * (1.01) * Z[:,:,:,2] 
    # else:
    #     Z[:,:,:,1] = 652 * Z[:,:,:,1]
    #     Z[:,:,:,2] = 800 * Z[:,:,:,2]
    
    #---------------------------------------------------
    # Saving Stuff 
    #---------------------------------------------------
    # Finding the values of everything (as percentages)
    # Decomp image 1:
    MSU_0_Seg_ST    = 100 * calculateMean(Z[:,:,:,0], ROI_array, square_size, 0, 3, 1, 3, 2, 3)
    MSU_0625_Seg_ST = 100 * calculateMean(Z[:,:,:,0], ROI_array, square_size, 0, 4, 1, 4, 2, 4)
    MSU_125_Seg_ST  = 100 * calculateMean(Z[:,:,:,0], ROI_array, square_size, 0, 5, 1, 5, 2, 5)
    MSU_25_Seg_ST   = 100 * calculateMean(Z[:,:,:,0], ROI_array, square_size, 0, 6, 1, 6, 2, 6)
    MSU_50_Seg_ST   = 100 * calculateMean(Z[:,:,:,0], ROI_array, square_size, 0, 7, 1, 7, 2, 7)
    HA_100_Seg_ST   = 100 * calculateMean(Z[:,:,:,0], ROI_array, square_size, 0, 0, 1, 0, 2, 0)
    HA_400_Seg_ST   = 100 * calculateMean(Z[:,:,:,0], ROI_array, square_size, 0, 1, 1, 1, 2, 1)
    HA_800_Seg_ST   = 100 * calculateMean(Z[:,:,:,0], ROI_array, square_size, 0, 2, 1, 2, 2, 2)
    
    # Decomp image 2:
    MSU_0_Seg_UA    = 100 * calculateMean(Z[:,:,:,1], ROI_array, square_size, 0, 3, 1, 3, 2, 3)
    MSU_0625_Seg_UA = 100 * calculateMean(Z[:,:,:,1], ROI_array, square_size, 0, 4, 1, 4, 2, 4)
    MSU_125_Seg_UA  = 100 * calculateMean(Z[:,:,:,1], ROI_array, square_size, 0, 5, 1, 5, 2, 5)
    MSU_25_Seg_UA   = 100 * calculateMean(Z[:,:,:,1], ROI_array, square_size, 0, 6, 1, 6, 2, 6)
    MSU_50_Seg_UA   = 100 * calculateMean(Z[:,:,:,1], ROI_array, square_size, 0, 7, 1, 7, 2, 7)
    HA_100_Seg_UA   = 100 * calculateMean(Z[:,:,:,1], ROI_array, square_size, 0, 0, 1, 0, 2, 0)
    HA_400_Seg_UA   = 100 * calculateMean(Z[:,:,:,1], ROI_array, square_size, 0, 1, 1, 1, 2, 1)
    HA_800_Seg_UA   = 100 * calculateMean(Z[:,:,:,1], ROI_array, square_size, 0, 2, 1, 2, 2, 2)
    
    # Decomp image 3:
    MSU_0_Seg_HA    = 100 * calculateMean(Z[:,:,:,2], ROI_array, square_size, 0, 3, 1, 3, 2, 3)
    MSU_0625_Seg_HA = 100 * calculateMean(Z[:,:,:,2], ROI_array, square_size, 0, 4, 1, 4, 2, 4)
    MSU_125_Seg_HA  = 100 * calculateMean(Z[:,:,:,2], ROI_array, square_size, 0, 5, 1, 5, 2, 5)
    MSU_25_Seg_HA   = 100 * calculateMean(Z[:,:,:,2], ROI_array, square_size, 0, 6, 1, 6, 2, 6)
    MSU_50_Seg_HA   = 100 * calculateMean(Z[:,:,:,2], ROI_array, square_size, 0, 7, 1, 7, 2, 7)
    HA_100_Seg_HA   = 100 * calculateMean(Z[:,:,:,2], ROI_array, square_size, 0, 0, 1, 0, 2, 0)
    HA_400_Seg_HA   = 100 * calculateMean(Z[:,:,:,2], ROI_array, square_size, 0, 1, 1, 1, 2, 1)
    HA_800_Seg_HA   = 100 * calculateMean(Z[:,:,:,2], ROI_array, square_size, 0, 2, 1, 2, 2, 2)

    MSU_0_Seg_dual      = 100 * calculateMean(Array_Dual, ROI_array, square_size, 0, 3, 1, 3, 2, 3)
    MSU_0625_Seg_dual   = 100 * calculateMean(Array_Dual, ROI_array, square_size, 0, 4, 1, 4, 2, 4)
    MSU_125_Seg_dual    = 100 * calculateMean(Array_Dual, ROI_array, square_size, 0, 5, 1, 5, 2, 5)
    MSU_25_Seg_dual     = 100 * calculateMean(Array_Dual, ROI_array, square_size, 0, 6, 1, 6, 2, 6)
    MSU_50_Seg_dual     = 100 * calculateMean(Array_Dual, ROI_array, square_size, 0, 7, 1, 7, 2, 7)

    print('*** Printing seg values as percentages:')
    print('MSU_0_Seg_ST: ' + str(MSU_0_Seg_ST))
    print('MSU_0625_Seg_ST: ' + str(MSU_0625_Seg_ST))
    print('MSU_125_Seg_ST: ' + str(MSU_125_Seg_ST))
    print('MSU_25_Seg_ST: ' + str(MSU_25_Seg_ST))
    print('MSU_50_Seg_ST: ' + str(MSU_50_Seg_ST))
    print('HA_100_Seg_ST: ' + str(HA_100_Seg_ST))
    print('HA_400_Seg_ST: ' + str(HA_400_Seg_ST))
    print('HA_800_Seg_ST: ' + str(HA_800_Seg_ST))
    print('MSU_0_Seg_UA: ' + str(MSU_0_Seg_UA))
    print('MSU_0625_Seg_UA: ' + str(MSU_0625_Seg_UA))
    print('MSU_125_Seg_UA: ' + str(MSU_125_Seg_UA))
    print('MSU_25_Seg_UA: ' + str(MSU_25_Seg_UA))
    print('MSU_50_Seg_UA: ' + str(MSU_50_Seg_UA))
    print('HA_100_Seg_UA: ' + str(HA_100_Seg_UA))
    print('HA_400_Seg_UA: ' + str(HA_400_Seg_UA))
    print('HA_800_Seg_UA: ' + str(HA_800_Seg_UA))
    print('MSU_0_Seg_HA: ' + str(MSU_0_Seg_HA))
    print('MSU_0625_Seg_HA: ' + str(MSU_0625_Seg_HA))
    print('MSU_125_Seg_HA: ' + str(MSU_125_Seg_HA))
    print('MSU_25_Seg_HA: ' + str(MSU_25_Seg_HA))
    print('MSU_50_Seg_HA: ' + str(MSU_50_Seg_HA))
    print('HA_100_Seg_HA: ' + str(HA_100_Seg_HA))
    print('HA_400_Seg_HA: ' + str(HA_400_Seg_HA))
    print('HA_800_Seg_HA: ' + str(HA_800_Seg_HA))
    print('MSU_0_Seg_dual: ' + str(MSU_0_Seg_dual))
    print('MSU_0625_Seg_dual: ' + str(MSU_0625_Seg_dual))
    print('MSU_125_Seg_dual: ' + str(MSU_125_Seg_dual))
    print('MSU_25_Seg_dual: ' + str(MSU_25_Seg_dual))
    print('MSU_50_Seg_dual: ' + str(MSU_50_Seg_dual))
    print()

    # Array2Copy = vtk.vtkImageData()
    # Array2Copy.DeepCopy(medicalImage)  
    
    # vtk_Array_UA = np.reshape(UA_Array, [dx * dy * dz], order='F')
    
    # vtk_data_array = numpy_to_vtk(vtk_Array_UA)
    
    # image = vtk.vtkImageData()
    
    # points = image.GetPointData()
    # points.SetScalars(vtk_data_array)
    
    # image = vtk.vtkImageData()
    # image.SetDimensions((dx, dy, dz))
    
    output = np.array([[Patient, Side, \
                        MSU_0_Seg_dual, MSU_0625_Seg_dual, MSU_125_Seg_dual, MSU_25_Seg_dual, MSU_50_Seg_dual, \
                        MSU_0_Seg_ST, MSU_0625_Seg_ST, MSU_125_Seg_ST, MSU_25_Seg_ST, MSU_50_Seg_ST, \
                        HA_100_Seg_ST, HA_400_Seg_ST,HA_800_Seg_ST,\
                        MSU_0_Seg_UA, MSU_0625_Seg_UA, MSU_125_Seg_UA, MSU_25_Seg_UA, MSU_50_Seg_UA, \
                        HA_100_Seg_UA, HA_400_Seg_UA,HA_800_Seg_UA,\
                        MSU_0_Seg_HA, MSU_0625_Seg_HA, MSU_125_Seg_HA, MSU_25_Seg_HA, MSU_50_Seg_HA, \
                        HA_100_Seg_HA, HA_400_Seg_HA,HA_800_Seg_HA,\
                        MSU_0_50keV, MSU_0625_50keV, MSU_125_50keV, MSU_25_50keV, MSU_50_50keV,\
                        HA_100_50keV, HA_400_50keV, HA_800_50keV,\
                        MSU_0_65keV, MSU_0625_65keV, MSU_125_65keV,MSU_25_65keV,MSU_50_65keV,\
                        HA_100_65keV, HA_400_65keV, HA_800_65keV]], dtype=object)
    # print(np.size(output))
    # if (write):
    #    print('Writing segementation to output file...')
     
    #   writer = vtk.vtkNIFTIImageWriter()
    #   writer.SetInputData(image)
    #   writer.SetFileName(os.path.join(outpath, 'Written.nii'))
        # copy most information directory from the header
        # writer.SetNIFTIHeader(reader50keV.GetNIFTIHeader())
        # this information will override the reader's header
        # writer.SetQFac(reader50keV.GetQFac())
        # writer.SetTimeDimension(reader50keV.GetTimeDimension())
        # writer.SetQFormMatrix(reader50keV.GetQFormMatrix())
        # writer.SetSFormMatrix(reader50keV.GetSFormMatrix())
    #    writer.Write()
    
    return output