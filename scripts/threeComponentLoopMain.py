#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#---------------------------------------------
# File: threeComponentLoopMain.py
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
import sys
import time
import numpy as np

from threeComponentDecomp import Component3_Decomp


# Need to add a option to extent your HA and UA inputs from the phantom so you can have above 1.. 
def main():
    # Directory structure that is assumed:
    #   Gout
    #    ├──scripts
    #    ├──output
    #    ├──input
    #    └──images
    #        ├──Gout01
    #        ├──Gout02
    #        ...
    #        └──GoutN
    #                ├──GoutN_L_50keV_1000 (DICOM directories assumed, must be named as shown)
    #                ├──GoutN_L_65keV_1001
    #                ├──GoutN_R_50keV_1002
    #                └──GoutN_R_65keV_1003
    #
    parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imageDir = os.path.join(parentDir, 'images')

    solvedArray = np.array([['Patient', 'Side', \
                            'MSU_0_Seg_dual', 'MSU_0625_Seg_dual', 'MSU_125_Seg_dual', 'MSU_25_Seg_dual', 'MSU_50_Seg_dual', \
                            'MSU_0_Seg_ST', 'MSU_0625_Seg_ST', 'MSU_125_Seg_ST', 'MSU_25_Seg_ST', 'MSU_50_Seg_ST', \
                            'HA_100_Seg_ST', 'HA_400_Seg_ST', 'HA_800_Seg_ST', \
                            'MSU_0_Seg_UA', 'MSU_0625_Seg_UA', 'MSU_125_Seg_UA', 'MSU_25_Seg_UA', 'MSU_50_Seg_UA', \
                            'HA_100_Seg_UA', 'HA_400_Seg_UA', 'HA_800_Seg_UA', \
                            'MSU_0_Seg_HA', 'MSU_0625_Seg_HA', 'MSU_125_Seg_HA', 'MSU_25_Seg_HA', 'MSU_50_Seg_HA', \
                            'HA_100_Seg_HA', 'HA_400_Seg_HA', 'HA_800_Seg_HA', \
                            'MSU_0_50', 'MSU_0625_50', 'MSU_125_50', 'MSU_25_50', 'MSU_50_50', \
                            'HA_100_50', 'HA_400_50', 'HA_800_50', \
                            'MSU_0_65', 'MSU_0625_65', 'MSU_125_65', 'MSU_25_65', 'MSU_50_65', \
                            'HA_100_65', 'HA_400_65', 'HA_800_65']], dtype=object)


    # Loop through the directories and perform decomposition on each patient
    pathL50keV = ''
    pathL65keV = ''
    pathR50keV = ''
    pathR65keV = ''
    patientNumber = 0
    sideL = ''
    sideR = ''
    solvedL = []
    solvedR = []

    for subDir1 in next(os.walk(imageDir))[1]:
        nextPatientDir = os.path.join(imageDir, subDir1)
        if 'phantom' in nextPatientDir.lower():
            continue

        for subDir2 in next(os.walk(nextPatientDir))[1]:
            nextImageDir = os.path.join(nextPatientDir, subDir2)

            # Get the path for the 50keV and 65keV images
            if '_L_' in nextImageDir:
                sideL = 'L'
                if '50keV' in nextImageDir:
                    pathL50keV = nextImageDir
                elif '65keV' in nextImageDir:
                    pathL65keV = nextImageDir
            elif '_R_' in nextImageDir:
                sideR = 'R'
                if '50keV' in nextImageDir:
                    pathR50keV = nextImageDir
                elif '65keV' in nextImageDir:
                    pathR65keV = nextImageDir

            # Get the patient number
            if subDir2[5] == '_':
                patientNumber = str(int(subDir2[4]))
            else:
                patientNumber = str(int(subDir2[4:6]))
        
        if patientNumber == 0:
            print('Error: could not find a valid patient number.')
            sys.exit(1)

        if sideL == 'L':
            if not pathL50keV or not pathL65keV:
                print('Error: could not find a filepath for left side.')
                sys.exit(1)
            
            print('*** Running 3MD using images:')
            print('50 keV: ' + str(pathL50keV))
            print('65 keV: ' + str(pathL65keV))
            print()
            solvedL = Component3_Decomp(pathL50keV, pathL65keV, patientNumber, 'L', filterimage=False, Correct_bone=True, Extend_seed=True)
            
            print('*** Printing 3 component decomposition for left side:')
            print(solvedL)
            print()
            
            solvedArray = np.vstack([solvedArray, solvedL])

        if sideR == 'R':
            if not pathR50keV or not pathR65keV:
                print('Error: could not find a filepath for right side.')
                sys.exit(1)
            
            print('*** Running 3MD using images:')
            print('50 keV: ' + str(pathR50keV))
            print('65 keV: ' + str(pathR65keV))
            print()
            solvedR = Component3_Decomp(pathR50keV, pathR65keV, patientNumber, 'R', filterimage=False, Correct_bone=True, Extend_seed=True)
            
            print('*** Printing 3 component decomposition for right side:')
            print(solvedR)
            print()
            
            solvedArray = np.vstack([solvedArray, solvedR])

    return solvedArray



if __name__ == '__main__':
    start_time = time.time()

    completed = main()
    completedString = completed.astype(str)
    completedString[1:,:] = completed[1:,:].astype('S7')

    patientNumber = completed[1][0]
    side = completed[1][1]

    # Write output to CSV
    parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputDir = os.path.join(parentDir, 'output')
    outputCSV = 'Patient_' + str(patientNumber) + '_Output.csv'
    outputFilename = os.path.join(outputDir, outputCSV)

    print('*** Writing outputs to CSV:')
    print(outputFilename)
    print()

    np.savetxt(outputFilename, completedString.astype(str), delimiter=',', fmt='%s')

    print('*** Done!')
    print('--- %s seconds ---' % (time.time() - start_time))