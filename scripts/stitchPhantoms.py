import os
import sys
import time
import argparse

import SimpleITK as sitk

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument( 'inputImage1', type=str, help='The first input image (path + filename)' )
parser.add_argument( 'inputImage2', type=str, help='The second input image (path + filename)' )
parser.add_argument( 'outputImage', type=str, help='The output image (path + filename)' )
args = parser.parse_args()

inputImage1Path = args.inputImage1
inputImage2Path = args.inputImage2
outputImagePath = args.outputImage

# Read in images (DICOM)
reader1 = sitk.ImageSeriesReader()
dicom_names = reader1.GetGDCMSeriesFileNames( inputImage1Path )
reader1.SetFileNames( dicom_names )
reader1.MetaDataDictionaryArrayUpdateOn()
reader1.LoadPrivateTagsOn()
topImage = reader1.Execute()
topImage.SetOrigin((0,0,0))

reader2 = sitk.ImageSeriesReader()
dicom_names = reader2.GetGDCMSeriesFileNames( inputImage2Path )
reader2.SetFileNames( dicom_names )
reader2.MetaDataDictionaryArrayUpdateOn()
reader2.LoadPrivateTagsOn()
bottomImage = reader2.Execute()
bottomImage.SetOrigin((0,0,0))

# Create an empty image of correct dimensions
widthList  = [ topImage.GetWidth(), bottomImage.GetWidth() ]
heightList = [ topImage.GetHeight(), bottomImage.GetHeight() ]

width  = max(widthList)
height = max(heightList)
depth  = bottomImage.GetDepth() + topImage.GetDepth()

emptyCombImage = sitk.Image(width, height, depth, bottomImage.GetPixelID())
emptyCombImage.SetSpacing(bottomImage.GetSpacing())
emptyCombImage.SetDirection(bottomImage.GetDirection())
emptyCombImage.SetOrigin([0, 0, 0])

# Add the images to the empty image
finalImage = sitk.Image(emptyCombImage.GetSize(), sitk.sitkInt32)
finalImage.SetOrigin([0,0,0])
finalImage.SetSpacing(emptyCombImage.GetSpacing())
finalImage.SetDirection(emptyCombImage.GetDirection())

# Paste in the top stack
pastedImg = sitk.Paste(finalImage, bottomImage, bottomImage.GetSize(), destinationIndex=[0, 0, 0])

# Paste in the top image (Z location is the top of the bottom image)
pastedZLocation = bottomImage.GetDepth()
pastedImg = sitk.Paste(pastedImg, topImage, topImage.GetSize(), destinationIndex=[0, 0, pastedZLocation])
print(pastedImg.GetSpacing())


# Write out the final image (DICOM)
# Create an output directory if it doesn't exist
if not os.path.exists(outputImagePath):
    try:
        os.mkdir(outputImagePath)
    except OSError as e:
        if e.errno != errno.EEXIST:     # Directory already exists error
            raise

writer = sitk.ImageFileWriter()
writer.KeepOriginalImageUIDOn()

imageTags = [ '0010|0010', # Patient Name
                '0010|0020', # Patient ID
                '0010|0030', # Patient Birth Date
                '0020|000D', # Study Instance UID, for machine consumption
                '0020|0010', # Study ID, for human consumption
                '0008|0020', # Study Date
                '0008|0030', # Study Time
                '0008|0050', # Accession Number
                '0008|0060',  # Modality
            ]

modificationTime = time.strftime('%H%M%S')
modificationDate = time.strftime('%Y%m%d')

seriesTagValues = [(k, reader1.GetMetaData(0,k)) for k in imageTags if reader1.HasMetaDataKey(0,k)] + \
                    [('0008|0031', modificationTime), # Series Time
                    ('0008|0021',  modificationDate), # Series Date
                    ('0008|0008',  'DERIVED\\SECONDARY'), # Image Type
                    ('0020|000e',  '1.2.826.0.1.3680043.2.1125.' + modificationDate + '.1' + modificationTime), # Series Instance UID
                    ('0020|0037',  '\\'.join(map(str, (pastedImg.GetDirection()[0], pastedImg.GetDirection()[3], pastedImg.GetDirection()[6], # Image Orientation (Patient)
                                                    pastedImg.GetDirection()[1], pastedImg.GetDirection()[4], pastedImg.GetDirection()[7])))),
                    ('0008|103e', reader1.GetMetaData(0, '0008|103e') + ' Processed-SimpleITK'), # Series Description
                    ('0028|0030', '\\'.join(map(str, (pastedImg.GetSpacing()[0], pastedImg.GetSpacing()[1]))))] # Spacing

for i in range(0, pastedImg.GetDepth()):
    imageSlice = pastedImg[:,:,i]

    # Tags shared by the series.
    for tag, value in seriesTagValues:
        imageSlice.SetMetaData(tag, value)
    
    # Slice specific tags.
    imageSlice.SetMetaData('0008|0012', time.strftime('%Y%m%d')) # Instance Creation Date
    imageSlice.SetMetaData('0008|0013', time.strftime('%H%M%S')) # Instance Creation Time
    imageSlice.SetMetaData('0020|0032', '\\'.join(map(str,pastedImg.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
    imageSlice.SetMetaData('0020|0013', str(i)) # Instance Number

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(os.path.join(outputImagePath, 'IM_' + str(i) + '.dcm'))
    writer.Execute(imageSlice)