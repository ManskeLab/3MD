#---------------------------------------------
# File: threeMaterialDecomp_MaskRemoval.py
#
# Created By: Justin J. Tse
# Created On: Oct 5, 2022
#
# Description: This function is to be performed after an initial 3MD. By utilizing one of the images as mask and 
#              removing said component from the initial source image for a second 3MD.
#   
#              Pseudo 4MD / 5MD
#
#
# Usage (hopefully): 
#              python threeMaterialDecomp_MaskRemoval.py LE.nii HE.nii MatX.nii
#              where MatX.nii = A, B, or C.nii generated from threeMaterialDecomp.py
#
#---------------------------------------------

import os
import argparse
from pathlib import Path
import SimpleITK as sitk



def threeMaterialDecomp_MaskRemoval(lowEnergyImagePath, highEnergyImagePath, maskImagePath, filterimage=False, Correct_bone=True, Extend_seed=True):
    parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Determines the path to one of the input images (LE) in this case
    # path.parent then will be the directory in which that image is located, useful for writing out the final images
    path = Path(lowEnergyImagePath)
#    print(path.parent)
#    outpath = os.path.join(parentDir, 'output')
    
    LE = sitk.ReadImage(lowEnergyImagePath, sitk.sitkFloat64)
    HE = sitk.ReadImage(highEnergyImagePath, sitk.sitkFloat64)
    Mask = sitk.ReadImage(maskImagePath, sitk.sitkFloat64)

    # creates inverted mask image
    percentmask = 1 - (Mask / 100)
    
    # multiples source images with inverted mask to remove masked decomp material depending
    # on its percent contribution per pixel
    LE_MaskRemoval = LE * percentmask
    HE_MaskRemoval = HE * percentmask
    
    # performed to acquire just the filename of the mask, use splitext to delineate between different words
    # and basename looks at the filename as a whole. [0] denotes whether you want the filename, while [1] would
    # give you the extension of the file
    maskname = os.path.splitext(os.path.basename(maskImagePath))[0]
    
    sitk.WriteImage(percentmask, os.path.join(path.parent, 'Inverted_' + maskname + '_Mask.nii'))
    sitk.WriteImage(LE_MaskRemoval, os.path.join(path.parent, 'LE_' + maskname + '_Removed.nii'))
    sitk.WriteImage(HE_MaskRemoval, os.path.join(path.parent, 'HE_' + maskname + '_Removed.nii'))
   
    return

if __name__ == '__main__':
    
    # Read in the input arguements
    parser = argparse.ArgumentParser(description='3MD Mask Removal')
    parser.add_argument('imageLowEnergy', help='The high energy image file path')
    parser.add_argument('imageHighEnergy', help='The low energy image file path')
    parser.add_argument('imageMask', help='The material parameters for decomposition (a)')
    args = parser.parse_args()

    # Parse arguments
    imageHighEnergy = args.imageHighEnergy
    imageLowEnergy = args.imageLowEnergy
    imageMask = args.imageMask

    threeMaterialDecomp_MaskRemoval(imageLowEnergy, imageHighEnergy, imageMask, filterimage=False, Correct_bone=True, Extend_seed=True)