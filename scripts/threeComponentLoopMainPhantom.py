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

from threeComponentDecompPhantom import Component3_Decomp


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

    solvedArray = np.array([['Patient', 'Leg', \
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

    for subDir1 in next(os.walk(imageDir))[1]:
        nextPatientDir = os.path.join(imageDir, subDir1)

        if not 'phantom' in nextPatientDir.lower():
            continue

        for subDir2 in next(os.walk(nextPatientDir))[1]:
            nextImageDir = os.path.join(nextPatientDir, subDir2)

            # Get the path for the 50keV and 65keV images
            if 'MSU_HA_PHANTOM_' in nextImageDir:
                if '65' in nextImageDir:
                    pathL65keV = nextImageDir
                elif '50' in nextImageDir:
                    pathL50keV = nextImageDir

            # Get the patient number
            # if subDir2[5] == '_':
            #     patientNumber = str(int(subDir2[4]))
            # else:
            #     patientNumber = 2
            patientNumber = str(0)

        if not pathL65keV or not pathL50keV or patientNumber == 0:
            print('Error: could not find a filepath, leg side, or patient number.')
            sys.exit(1)
            
        print('pathL50keV: ' + str(pathL50keV))
        print('pathL65keV: ' + str(pathL65keV))
        solvedArrayPhantom = Component3_Decomp(pathL50keV, pathL65keV, patientNumber, 'L', filterimage=False, Correct_bone=True, Extend_seed=True)

        solvedArray = np.vstack([solvedArray, solvedArrayPhantom])

        print()
        print('Printing 3 component decomposition for phantom:')
        print(solvedArray)
        print()

    return solvedArray



if __name__ == '__main__':
    start_time = time.time()

    completed = main()
    completedString = completed.astype(str)
    completedString[1:,:] = completed[1:,:].astype('S7')

    patientNumber = completed[0]

    # Write output to CSV
    parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputDir = os.path.join(parentDir, 'output')
    outputFilename = os.path.join(outputDir, 'Output_Phantom.csv')
    np.savetxt(outputFilename, completedString.astype(str), delimiter=',', fmt='%s')

    print('--- %s seconds ---' % (time.time() - start_time))