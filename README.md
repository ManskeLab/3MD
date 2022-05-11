# Manske Lab 3MD

This repository contains scripts for the manuscript:

***J.J. Tse, D.A. Kondro, M.T. Kuczynski, Y. Pauchard, A. Veljkovic, D.W. Holdsworth, V. Frasson, S.L. Manske, P. MacMullan, P. Salat. Assessing the Sensitivity of Dual-Energy Computed Tomography 3-Material Decomposition for the Detection of Gout. Investigative Radiology: April 22, 2022. In Press. DOI: 10.1097/RLI.0000000000000879***

## Requirements:
1. Python (v3.8.5)
2. Numpy (v1.20.3)
3. SimpleITK (v2.0.2)
4. VTK (v9.0.1)

## How to run:
1. Save your input CSV to the *input* directory
2. Run the *threeComponentLoopMain.py* script in the *scripts* directory
3. Outputs will be saved to the *output* directory with a separate folder for each patient

## Assumed directory structure and file naming convention:
```
Gout
  ├──scripts
  ├──output
  ├──input
  └──images
      ├──Gout01
      ├──Gout02
     ...
      └──GoutN
            ├──GoutN_L_50keV_1000 (DICOM directories assumed, must be named as shown)
            ├──GoutN_L_65keV_1001
            ├──GoutN_R_50keV_1002
            └──GoutN_R_65keV_1003
```
