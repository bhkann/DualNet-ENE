## Preprocessing Code for Deep Learning Head and Neck Project (c) Written by Benjamin H. Kann, Revised 8/16/2022 ##
## Requires two saved nrrd files: 1) image file, and 2) segmentation (lymph node) 3D mask

import os
import sys
import math
import re
import shutil
import glob

import pandas as pd
import numpy as np

import scipy
from scipy.spatial import ConvexHull
from scipy import ndimage
from scipy.ndimage import binary_dilation
import SimpleITK as sitk

sys.path.append('~/DualNet-ENE/utils')
from dcm_to_nrrd import dcm_to_nrrd
from interpolate import interpolate
from crop_roi import crop_roi_ene

##############################################################################################################################################################
############################## MAIN CODE #####################################################################################################
##############################################################################################################################################################
#Hardcoded specifications ##
img_spacing = (1,1,3) # Original Model Spacing
crop_shape = (118,118,32) # Shape of Original BoxNet
data_type = 'ct'

#User-Defined Variables (can be modified as needed):
INPUT_PATH = '~/DualNet-ENE/ENE_preprocess_nrrd' # Path where two nrrds are saved. Image file must contain "CT" suffix; mask file must contain "mask3d" suffix

INTERP_DIR = '~/DualNet-ENE/interpolated' + ''.join(str(img_spacing)) # Directory to save interpolated nrrds;

OUTPUT_IMAGE_DIR = '/DualNet-ENE/image_crop' + ''.join(str(spacing))
OUTPUT_MASK_DIR = '/DualNet-ENE/label_crop' + ''.join(str(spacing))

dataset = 'TestCases' # Label for the test dataset

try:
    os.mkdir(INTERP_DIR)
except:
    print("directory already exists")


for file in sorted(os.listdir(INPUT_PATH)): #LOOP Goes through nrrd raw images etc
    #IMAGE INTERPOLATION
    if 'CT' in file:
        patient_id = '-'.join(file.split('_')[1:3])
        print("patient ID: ", patient_id)
        path_to_nrrd = os.path.join(INPUT_PATH,file)            
        interpolation_type = "linear"
        return_type = "numpy_array"
        interpolate(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, img_spacing, return_type, INTERP_DIR)
        print("image interpolated")
    
    #MASK INTERPOLATION
    if 'mask3d' in file:
        roi_id = patient_id + '_' + file.replace('.','_').split('_')[3] ## If more than one segmentation in same patient scan
        print("patient ID: ", patient_id)          
        interpolation_type = "nearest_neighbor" #"linear" for image, nearest neighbor for label
        return_type = "sitk_object"
        interpolated_img = interpolate(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, img_spacing, return_type, output_dir="")
        print("label interpolated")
        interpolated_array = sitk.GetArrayFromImage(interpolated_img)
        
        #Mask dilation to encompass surrounding tissue
        dilated_array = binary_dilation(interpolated_array, structure=np.ones((1,5,5),np.int32), iterations=2, mask=None, output=None, border_value=0, origin=0, brute_force=False).astype(interpolated_array.dtype)    
        dilated_img = sitk.GetImageFromArray(dilated_array)
        dilated_img.SetSpacing(interpolated_img.GetSpacing())
        dilated_img.SetDirection(interpolated_img.GetDirection())
        dilated_img.SetOrigin(interpolated_img.GetOrigin())
        nrrd_file_path = INTERP_DIR + '/' + roi_id + '_interpolated.nrrd'
        nrrdWriter = sitk.ImageFileWriter()
        nrrdWriter.SetFileName(nrrd_file_path)
        nrrdWriter.SetUseCompression(True)
        nrrdWriter.Execute(dilated_img)
        print("nrrd saved")

### Center and Crop 
try:
    os.mkdir(OUTPUT_IMAGE_DIR)
    os.mkdir(OUTPUT_MASK_DIR)
except:
    print("directory already exists")

return_type='numpy_array'
for file in sorted(os.listdir(INTERP_DIR)):
    if 'ct' not in file:
        label_id = '_'.join(file.split('_')[:2])
        patient_id = label_id.split('_')[0]
        print('label',label_id, 'patient', patient_id)
        path_to_label_nrrd = os.path.join(INTERP_DIR,file)
        path_to_image_nrrd = os.path.join(INTERP_DIR,dataset + '_' + patient_id + '_ct_interpolated_raw_raw_xx.nrrd')
        image_obj, label_obj = crop_roi_ene(dataset,
                                        label_id,
                                        label_id,
                                        path_to_image_nrrd, 
                                        path_to_label_nrrd, 
                                        crop_shape,
                                        return_type, 
                                        OUTPUT_IMAGE_DIR, 
                                        OUTPUT_MASK_DIR)