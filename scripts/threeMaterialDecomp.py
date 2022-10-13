#---------------------------------------------
# File: threeMaterialDecomp.py
#
# Created By: Michael Kuczynski
# Created On: July 8, 2022
#
# Description:
#
# Usage:
#   python threeMaterialDecomp.py imageHighEnergy.nii imageLowEnergy.nii
#   python threeMaterialDecomp.py imageHighEnergy.nii imageLowEnergy.nii a b c e f g
#
#
# Modified by: Justin J. Tse
# Modified On: Oct 4, 2022
# Description: Added in fixes to reorient images after the VTK --> NP --> VTK conversion
#---------------------------------------------

import os
import vtk
import sys
import time
import argparse
import numpy as np
import SimpleITK as sitk
from fileConverter import fileConverter
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def project_precent_a_array(point_x, point_y, a_x, a_y, b_x, b_y):
    # Has errors!!!!!!
    # We find the line between the two input points
    m = (b_y - a_y) / (b_x - a_x)
    b = b_y - (m * b_x)

    # Our projection must be an orthogonal line to this
    m_orthogonal = -1/m
    b_orthogonal = point_y - (m_orthogonal * point_x)

    # Find where they intercept
    x_intercept = (b_orthogonal - b) / (m - m_orthogonal)
    y_intercept = (m_orthogonal * x_intercept) + b_orthogonal

    # Find the distances
    a_2_b = np.sqrt((a_x - b_x)**2 + (a_y - b_y)**2)
    b_2_intercept = np.sqrt((b_x - x_intercept)**2 + (b_y - y_intercept)**2)
    a_2_intercept = np.sqrt((a_x - x_intercept)**2 + (a_y - y_intercept)**2)
    percent_a = b_2_intercept / a_2_b

    # If our x point is to the right point b and the slope is positive and point a is the lower point set to 1
    percent_a[b_2_intercept > a_2_b] = 1 # Only works if the distance is not outside of length from a to b

    percent_a[a_2_intercept > a_2_b] = 0
    if (np.sum(percent_a > 1)) > 0:
        print('There are some precentages greater than 1')


    return percent_a


def extend_x_y(a_x, a_y, b_x, b_y, extension=1):
    # point x and point y are arrays
    c_x = (b_x - a_x) * (extension + 1) + a_x
    c_y = (b_y - a_y) * (extension + 1) + a_y
    
    return c_x, c_y


def project_precent_a_array_old(point_x, point_y, a_x, a_y, b_x, b_y):
    # Not the best but it works... 
    # point x and point y are arrays
    point_2_a = np.sqrt((point_x - a_x)**2 + (point_y - a_y)**2)
    point_2_b = np.sqrt((point_x - b_x)**2 + (point_y - b_y)**2)
    a_2_b = np.sqrt((a_x - b_x)**2 + (a_y - b_y)**2)

    s = (point_2_a + point_2_b + a_2_b) / 2 # Herons Formula to find area
    A = np.sqrt(s*(s - point_2_a)*(s - point_2_b)*(s - a_2_b)) # Herons Formula to find area
    h = 2 * A / a_2_b  # 1/2 base * height = Area
    project_2_a = np.sqrt(point_2_a**2 - h**2)
    project_2_b = np.sqrt(point_2_b**2 - h**2)
    percent_a = project_2_b / a_2_b

    percent_a[percent_a > 1] = 1
    percent_a[percent_a < 0] = 0
        
    return percent_a


def calculateMean(array, ROI_array, square_size, a, b, c, d, e, f):
    mean = (array[int(ROI_array[a,b] - square_size/2):int(ROI_array[a,b] + square_size/2), \
                    int(ROI_array[c,d] - square_size/2):int(ROI_array[c,d] + square_size/2), \
                    int(ROI_array[e,f] - square_size/2):int(ROI_array[e,f] + square_size/2)])
    return mean


def threeMaterialDecomp(lowEnergyImagePath, highEnergyImagePath, a=0, b=0, c=0, e=0, f=0, g=0, filterimage=False, Correct_bone=True, Extend_seed=True):
    parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    path = Path(lowEnergyImagePath)
    outpath = os.path.join(parentDir, 'output')

    # Load the low and high energy images (Must be DICOM!)
    reader50keV = vtk.vtkDICOMImageReader() 
    reader50keV.SetDirectoryName(lowEnergyImagePath) 
    reader50keV.SetDataScalarTypeToUnsignedShort()
    reader50keV.UpdateWholeExtent()
    reader50keV.Update()
    
    # Obtains the native coordinates of the data
    reader50keV.GetImagePositionPatient()
   
    reader65keV = vtk.vtkDICOMImageReader() 
    reader65keV.SetDirectoryName(highEnergyImagePath) 
    reader65keV.SetDataScalarTypeToUnsignedShort()
    reader65keV.UpdateWholeExtent()
    reader65keV.Update()
    reader65keV.GetImagePositionPatient()



    
    (xsize, ysize, zsize) = reader50keV.GetOutput().GetDimensions()

    # print('Should we filter?')
    if filterimage:
        # print('Applying Gaussian Smooth')
        filter_image = vtk.vtkImageGaussianSmooth()
        filter_image.SetInputConnection(reader50keV.GetOutputPort())
        filter_image.SetStandardDeviation(2)
        filter_image.SetRadiusFactors(3,3,3)
        filter_image.SetDimensionality(3)
        filter_image.Update()
        nodes_vtk_array_50 = filter_image.GetOutput().GetPointData().GetArray(0)

        filter_image.SetInputConnection(reader65keV.GetOutputPort())
        filter_image.Update()
        nodes_vtk_array_65 = filter_image.GetOutput().GetPointData().GetArray(0)
    else:
        nodes_vtk_array_50 = reader50keV.GetOutput().GetPointData().GetScalars()
        nodes_vtk_array_65 = reader65keV.GetOutput().GetPointData().GetScalars()

    image_array_50keV = vtk_to_numpy(nodes_vtk_array_50)
    image_array_65keV = vtk_to_numpy(nodes_vtk_array_65)
    
    Array_65keV = np.reshape(image_array_65keV, [xsize, ysize, zsize], order='F')
    Array_50keV = np.reshape(image_array_50keV, [xsize, ysize, zsize], order='F')
    Array_Dual = Array_65keV
    Array_Three = Array_65keV
    

    # Doing it without for loops to speed things up by around 40 minutes
    if a | b | c | e | f | g == 0:
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
        ROI_array = ROI_array[ROI_array[:,1]==Leg,:] # Same as above but for right or left leg
        ROI_array = ROI_array[:,3:-1] # Slice out first two columns and last column. Not sure what is in these columns...
        ROI_array = ROI_array.astype(float)

        print()
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
        sys.exit()

        #------------------------------------------------------
        # Applying the Threshold 
        #------------------------------------------------------
        Array_Dual = Array_65keV
        
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
        b_line = y_point-m * x_point
        
        Array_Dual = 1*(image_array_65keV > image_array_50keV * m + b_line) + 1*(image_array_65keV > horizontal_50keV) > 1
        Array_Dual = np.reshape(Array_Dual,[xsize, ysize, zsize], order='F')

        #------------------------------------------------------
        # Applying the Three Component Analysis 
        #------------------------------------------------------
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
        
        # print('Applying 3 component decomp')
        if Extend_seed:
            UA_50keV, UA_65keV = extend_x_y(MSU_0_50keV, MSU_0_65keV, MSU_50_50keV, MSU_50_65keV, extension=1) 
            HA_50keV, HA_65keV = extend_x_y(HA_100_50keV, HA_100_65keV, HA_800_50keV, HA_800_65keV, extension=0.5)  

        # print('Solving the linear equations')
        ST_50keV = float(ST_50keV)
        ST_65keV = float(ST_65keV)
        
        UA_50keV = float(UA_50keV)
        UA_65keV = float(UA_65keV)
        
        HA_50keV = float(HA_50keV)
        HA_65keV = float(HA_65keV)

        a = ST_50keV
        b = UA_50keV
        c = HA_50keV
        e = ST_65keV
        f = UA_65keV
        g = HA_65keV
    else:
        ST_50keV = a
        UA_50keV = b
        HA_50keV = c
        ST_65keV = e
        UA_65keV = f
        HA_65keV = g
    d = Array_50keV
    h = Array_65keV  

    Z = np.zeros((xsize, ysize, zsize,3))                
    Z[:,:,:,1] = ( (d-a)-(h-e) * (c-a) / (g-e) ) / ( (b-a)-(c-a) * (f-e) / (g-e) ) 
    Z[:,:,:,2] = ( (d-a)-(h-e) * (b-a) / (f-e) ) / ( (c-a)-(b-a) * (g-e) / (f-e) )
    Z[:,:,:,0] = 1 - Z[:,:,:,2] - Z[:,:,:,1]
    
    # Z_hold1
    Z = Z.astype(float)
    Array_50keV = Array_50keV.astype(float)
    
    # Removing Air
    print('np.average(Z[:,:,:,0] + Z[:,:,:,1] + Z[:,:,:,2]): ' + str(np.average(Z[:,:,:,0] + Z[:,:,:,1] + Z[:,:,:,2])))
    
    Z[Array_50keV<-300.00,:] = 0

    # print('Removing the double negatives')
    # Everything with a double negative is kept
    Z_DN = ((Z[:,:,:,0] * Z[:,:,:,1] * Z[:,:,:,2]) > 0) & ((np.absolute(Z[:,:,:,0]) + np.absolute(Z[:,:,:,1]) + np.absolute(Z[:,:,:,2])) > 1)
    Z_S = ((Z[:,:,:,0] * Z[:,:,:,1] * Z[:,:,:,2]) > 0) & ((np.absolute(Z[:,:,:,0]) + np.absolute(Z[:,:,:,1]) + np.absolute(Z[:,:,:,2])) == 1)

    print('np.sum(Z_DN): ' + str(np.sum(Z_DN)))
    
    # And set the remaining positive to 1 if it had a double negative
    Z[(Z_DN) & (Z[:,:,:,0]>1.00),0] = 1.00
    Z[(Z_DN) & (Z[:,:,:,1]>1.00),1] = 1.00
    Z[(Z_DN) & (Z[:,:,:,2]>1.00),2] = 1.00
    Z[Z<0] = 0

    # print('Finding points to be interpolated')
    # Itentify the points that need to be interpolated, the ones with two numbers!
    Z_TS = ((Z[:,:,:,1] + Z[:,:,:,2]) > 1.00) & (Z[:,:,:,1] * Z[:,:,:,2] > 0)
    Z_UA = ((Z[:,:,:,0] + Z[:,:,:,2]) > 1.00) & (Z[:,:,:,0] * Z[:,:,:,2] > 0)
    Z_HA = ((Z[:,:,:,0] + Z[:,:,:,1]) > 1.00) & (Z[:,:,:,0] * Z[:,:,:,1] > 0)
    
    print('np.sum(Z_TS * Z_DN): ' + str(np.sum(Z_TS * Z_DN)))
    print('np.sum(Z_UA * Z_DN): ' + str(np.sum(Z_UA * Z_DN)))
    print('np.sum(Z_HA * Z_DN): ' + str(np.sum(Z_HA * Z_DN)))
    print('np.sum(Z_HA * Z_UA): ' + str(np.sum(Z_HA * Z_UA)))
    print('np.sum(Z_HA * Z_TS): ' + str(np.sum(Z_HA * Z_TS)))
    print('np.sum(Z_UA * Z_TS): ' + str(np.sum(Z_UA * Z_TS)))
    print('np.average(Z_DN | Z_HA |  Z_UA |  Z_TS | Z_S | (Array_50keV<-300.00)): ' + str(np.average(Z_DN | Z_HA |  Z_UA |  Z_TS | Z_S | (Array_50keV<-300.00))))
    print('np.max(Z[:,:,0]): ' + str(np.max(Z[:,:,0])))
    print('np.max(Z[:,:,:,1]): ' + str(np.max(Z[:,:,:,1])))
    print('np.max(Z[:,:,2]): ' + str(np.max(Z[:,:,2])))
    print('np.min(Z[:,:,0]): ' + str(np.min(Z[:,:,0])))
    print('np.min(Z[:,:,:,1]): ' + str(np.min(Z[:,:,:,1])))
    print('np.min(Z[:,:,2]): ' + str(np.min(Z[:,:,2])))

    # print('Finding the zero TSs')
    Z[Z_TS,1] = project_precent_a_array(Array_50keV[Z_TS], Array_65keV[Z_TS], UA_50keV, UA_65keV, HA_50keV, HA_65keV)
    Z[Z_TS,2] = 1 - Z[Z_TS,1]
    
    # print('Finding the zero UAs')
    Z[Z_UA,0] = project_precent_a_array(Array_50keV[Z_UA], Array_65keV[Z_UA], ST_50keV, ST_65keV, HA_50keV, HA_65keV)
    Z[Z_UA,2] = 1 - Z[Z_UA,0]
    
    # print('Finding the zero HAs')
    Z[Z_HA,0] = project_precent_a_array(Array_50keV[Z_HA], Array_65keV[Z_HA], ST_50keV, ST_65keV, UA_50keV, UA_65keV)
    Z[Z_HA,1] = 1 - Z[Z_HA,0]
    
    print('np.max(Z[:,:,0]): ' + str(np.max(Z[:,:,0])))
    print('np.max(Z[:,:,:,1]): ' + str(np.max(Z[:,:,:,1])))
    print('np.max(Z[:,:,2]): ' + str(np.max(Z[:,:,2])))
    print('np.min(Z[:,:,0]): ' + str(np.min(Z[:,:,0])))
    print('np.min(Z[:,:,:,1]): ' + str(np.min(Z[:,:,:,1])))
    print('np.min(Z[:,:,2]): ' + str(np.min(Z[:,:,2])))

    # Save the 3D images
    # We should have 7 images total
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # im = sitk.GetImageFromArray(np.rot90((Z[:,:,:,0] * 255).astype(np.uint8)))
    # sitk.WriteImage(im, os.path.join(outpath, 'A.nii'))
    # im = sitk.GetImageFromArray(np.rot90((Z[:,:,:,1] * 255).astype(np.uint8)))
    # sitk.WriteImage(im, os.path.join(outpath, 'B.nii'))  
    # im = sitk.GetImageFromArray(np.rot90((Z[:,:,:,2] * 255).astype(np.uint8)))
    # sitk.WriteImage(im, os.path.join(outpath, 'C.nii'))
    # im = sitk.GetImageFromArray(np.rot90((((Array_50keV[:,:,:] - np.min(Array_50keV)) / (np.max(Array_50keV) - np.min(Array_50keV))) * 255).astype(np.uint8)))
    # sitk.WriteImage(im, os.path.join(outpath, '50keV.nii'))
    # im = sitk.GetImageFromArray(np.rot90((((Array_65keV[:,:,:] - np.min(Array_65keV)) / (np.max(Array_65keV) - np.min(Array_65keV))) * 255).astype(np.uint8)))
    # sitk.WriteImage(im, os.path.join(outpath, '65keV.nii'))
    # im = sitk.GetImageFromArray(np.rot90((Array_Dual[:,:,:] * 255).astype(np.uint8)))
    # sitk.WriteImage(im, os.path.join(outpath, 'Dual.nii'))
    # im2 = sitk.InvertIntensity(im)
    # sitk.WriteImage(im2, os.path.join(outpath, 'Dual_Sub.nii'))

    
    #------------------------------------------------------
    # Attempting to write out vtk --> np --> vtk while retaining orientation
    #------------------------------------------------------
    
    
    # Creates empty arrays for decomp materials A, B, and C based on the image dimensions of the 50keV source image
    # For Decomp Material A
    new_vtk_image_MatA = vtk.vtkImageData()
    new_vtk_image_MatA.DeepCopy(reader50keV.GetOutput())
    new_vtk_image_MatA.GetPointData().SetScalars(numpy_to_vtk((Z[:,:,:,0] * 100).ravel(order='F'),deep=True))
    
    
    resliceMatA = vtk.vtkImageReslice()
    resliceMatA.SetInputData(new_vtk_image_MatA)
    resliceMatA.SetInterpolationModeToLinear()
    resliceMatA.SetResliceAxesDirectionCosines([-1, 0, 0, 0, 1, 0, 0, 0, -1])
    resliceMatA.Update()
    reslice_img_MatA = resliceMatA.GetOutput()
    
    # Orientates the decomp material into the same origin as the source 50keV image
    reslice_img_MatA.SetOrigin(reader50keV.GetImagePositionPatient())
    
    writerMatA = vtk.vtkNIFTIImageWriter()
    writerMatA.SetInputData(reslice_img_MatA)
    writerMatA.SetFileName(os.path.join(path.parent, 'A.nii'))
    writerMatA.Write()
    
    # For Decomp Material B
    
    new_vtk_image_MatB = vtk.vtkImageData()
    new_vtk_image_MatB.DeepCopy(reader50keV.GetOutput())
    new_vtk_image_MatB.GetPointData().SetScalars(numpy_to_vtk((Z[:,:,:,1] * 100).ravel(order='F'),deep=True))
    
    resliceMatB = vtk.vtkImageReslice()
    resliceMatB.SetInputData(new_vtk_image_MatB)
    resliceMatB.SetInterpolationModeToLinear()
    resliceMatB.SetResliceAxesDirectionCosines([-1, 0, 0, 0, 1, 0, 0, 0, -1])
    resliceMatB.Update()
    reslice_img_MatB = resliceMatB.GetOutput()
    
    # Orientates the decomp material into the same origin as the source 50keV image
    reslice_img_MatB.SetOrigin(reader50keV.GetImagePositionPatient())
    
    writerMatB = vtk.vtkNIFTIImageWriter()
    writerMatB.SetInputData(reslice_img_MatB)
    writerMatB.SetFileName(os.path.join(path.parent, 'B.nii'))
    writerMatB.Write()
    
     # For Decomp Material C
    
    new_vtk_image_MatC = vtk.vtkImageData()
    new_vtk_image_MatC.DeepCopy(reader50keV.GetOutput())
    new_vtk_image_MatC.GetPointData().SetScalars(numpy_to_vtk((Z[:,:,:,2] * 100).ravel(order='F'),deep=True))
    
    resliceMatC = vtk.vtkImageReslice()
    resliceMatC.SetInputData(new_vtk_image_MatC)
    resliceMatC.SetInterpolationModeToLinear()
    resliceMatC.SetResliceAxesDirectionCosines([-1, 0, 0, 0, 1, 0, 0, 0, -1])
    resliceMatC.Update()
    reslice_img_MatC = resliceMatC.GetOutput()
    
    # Orientates the decomp material into the same origin as the source 50keV image
    reslice_img_MatC.SetOrigin(reader50keV.GetImagePositionPatient())
    
    writerMatC = vtk.vtkNIFTIImageWriter()
    writerMatC.SetInputData(reslice_img_MatC)
    writerMatC.SetFileName(os.path.join(path.parent, 'C.nii'))
    writerMatC.Write()   
    
    # For Decomp Material 50 and 65 keV source images
    
#     fileConverter(lowEnergyImagePath, os.path.join(outpath, 'LE.nii'))
#     fileConverter(highEnergyImagePath, os.path.join(outpath, 'HE.nii'))
        
    
    new_vtk_image_50keV = vtk.vtkImageData()
    new_vtk_image_50keV.DeepCopy(reader50keV.GetOutput())
    new_vtk_image_50keV.GetPointData().SetScalars(numpy_to_vtk((Array_50keV[:,:,:]).ravel(order='F'),deep=True))
    
    reslice50keV = vtk.vtkImageReslice()
    reslice50keV.SetInputData(new_vtk_image_50keV)
    reslice50keV.SetInterpolationModeToLinear()
    reslice50keV.SetResliceAxesDirectionCosines([-1, 0, 0, 0, 1, 0, 0, 0, -1])
    reslice50keV.Update()
    reslice_img_50keV = reslice50keV.GetOutput()
    
    # Orientates the decomp material into the same origin as the source 50keV image
    reslice_img_50keV.SetOrigin(reader50keV.GetImagePositionPatient())
    
    writer50keV = vtk.vtkNIFTIImageWriter()
    writer50keV.SetInputData(reslice_img_50keV)
    writer50keV.SetFileName(os.path.join(path.parent, 'LE.nii'))
    writer50keV.Write()   
    
    
    new_vtk_image_65keV = vtk.vtkImageData()
    new_vtk_image_65keV.DeepCopy(reader50keV.GetOutput())
    new_vtk_image_65keV.GetPointData().SetScalars(numpy_to_vtk((Array_65keV[:,:,:]).ravel(order='F'),deep=True))
    
    reslice65keV = vtk.vtkImageReslice()
    reslice65keV.SetInputData(new_vtk_image_65keV)
    reslice65keV.SetInterpolationModeToLinear()
    reslice65keV.SetResliceAxesDirectionCosines([-1, 0, 0, 0, 1, 0, 0, 0, -1])
    reslice65keV.Update()
    reslice_img_65keV = reslice65keV.GetOutput()
    
    # Orientates the decomp material into the same origin as the source 50keV image
    reslice_img_65keV.SetOrigin(reader65keV.GetImagePositionPatient())
    
    writer65keV = vtk.vtkNIFTIImageWriter()
    writer65keV.SetInputData(reslice_img_65keV)
    writer65keV.SetFileName(os.path.join(path.parent, 'HE.nii'))
    writer65keV.Write()   
    
    
    
    return



if __name__ == '__main__':
    start_time = time.time()

    # Read in the input arguements
    parser = argparse.ArgumentParser(description='3 Material Decomposition - General Script')
    parser.add_argument('imageHighEnergy', help='The high energy image file path')
    parser.add_argument('imageLowEnergy', help='The low energy image file path')
    parser.add_argument('a', nargs='?', type=int, default=0, help='The material parameters for decomposition (a)')
    parser.add_argument('b', nargs='?', type=int, default=0, help='The material parameters for decomposition (b)')
    parser.add_argument('c', nargs='?', type=int, default=0, help='The material parameters for decomposition (c)')
    parser.add_argument('e', nargs='?', type=int, default=0, help='The material parameters for decomposition (e)')
    parser.add_argument('f', nargs='?', type=int, default=0, help='The material parameters for decomposition (f)')
    parser.add_argument('g', nargs='?', type=int, default=0, help='The material parameters for decomposition (g)')
    args = parser.parse_args()

    # Parse arguments
    imageHighEnergy = args.imageHighEnergy
    imageLowEnergy = args.imageLowEnergy
    a = args.a
    b = args.b
    c = args.c
    e = args.e
    f = args.f
    g = args.g

    threeMaterialDecomp(imageLowEnergy, imageHighEnergy, a, b, c, e, f, g, filterimage=False, Correct_bone=True, Extend_seed=True)

    print('--- %s seconds ---' % (time.time() - start_time))