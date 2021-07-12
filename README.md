# Gout

## How to run:
1. Save your input CSV to the *input* directory
1. Activate the manskelab conda environment (see Manskelab/Manskelab on GitHub for instructions)
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