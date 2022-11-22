## Preprocessing Code for Deep Learning Head and Neck Project (c) Written by Benjamin H. Kann, Revised 8/16/2022 ##

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


## Bring in Images/Labels (from 3D Slicer annotation)
path_input = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd'
path_new_image =  '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/new_raw_images'
path_new_label = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/new_raw_labels'
for file in sorted(os.listdir(path_new_label)):
    scan_name = '_'.join(file.split('_')[0:2])
    node_name = '_'.join(re.split(', |_|-|\.', file)[0:3])
    src_file = os.path.join(path_new_label,file)
    dst_file = os.path.join(path_input,'mask3d_' + node_name + '.nrrd')
    dst_image = os.path.join(path_input,'E3311_' + scan_name + '_CT_raw_raw_raw_xx.nrrd')
    print(src_file,dst_file,dst_image)
    shutil.copy(src_file,dst_file)
    shutil.copy(glob(path_new_image + '/*' + scan_name + '*')[0],dst_image)

## INTERPOLATION ##
spacing = (1,1,3) #(0.75,0.75,2.5)
INTERP_DIR = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/interpolated' + ''.join(str(spacing))
try:
    os.mkdir(INTERP_DIR)
except:
    print("directory already exists")

img_spacing = spacing
path_input = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd'
dataset = 'E3311'
for file in sorted(os.listdir(path_input)): #LOOP Goes through nrrd raw images etc
    if not file.startswith('.') and 'CT' in file:
        patient_id = '-'.join(file.split('_')[1:3])
        print("patient ID: ", patient_id)
        path_to_nrrd = os.path.join(path_input,file)            
        print("image path: ",path_to_nrrd)   
        data_type = "ct" #ct or pet
        # path_to_nrrd = "/data/output/dataset_124_ct_raw_raw_raw_xx.nrrd"
        if spacing[2] == 9:
            sitk_img = sitk.ReadImage(path_to_nrrd)
            img_spacing = list(sitk_img.GetSpacing())
            ## NEED TO FINISH : NEED TO APPEND THE ACTUAL Z to tHE DEFINED X,Y
            img_spacing = tuple(np.append(list(spacing[:2]),img_spacing[2]))
            INTERP_DIR = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/interpolated' + ''.join(str(spacing))
            try:
                os.mkdir(INTERP_DIR)
            except:
                print("directory already exists")
        interpolation_type = "linear" #"linear" for image, nearest neighbor for label
        #spacing = (1,1,3) # x,y,z For HN SEG: 1, 1, 3; for ENE input: 0.75, 0.75, ?unchanged?2.5?
        return_type = "numpy_array"
        #output_dir = "/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/interpolated"
        interpolate(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, img_spacing, return_type, INTERP_DIR)
        print("image interpolated")
    if not file.startswith('.') and 'mask3d' in file:
        patient_id = '-'.join(file.split('_')[1:3])
        roi_id = patient_id + '_' + file.replace('.','_').split('_')[3]
        print("patient ID: ", patient_id)
        path_to_nrrd = os.path.join(path_input,file)            
        print("image path: ",path_to_nrrd)   
        data_type = "ct" #ct or pet
        # path_to_nrrd = "/data/output/dataset_124_ct_raw_raw_raw_xx.nrrd"
        sitk_label = sitk.ReadImage(path_to_nrrd)
        #spacing_label = sitk_label.GetSpacing()
        interpolation_type = "nearest_neighbor" #"linear" for image, nearest neighbor for label
        #spacing = (1,1,3) # x,y,z For HN SEG: 1, 1, 3; for ENE input: 0.75, 0.75, ?unchanged?
        return_type = "sitk_object"
        #output_dir = "/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/interpolated"
        interpolated_img = interpolate(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, img_spacing, return_type, output_dir="")
        print("label interpolated")
        interpolated_array = sitk.GetArrayFromImage(interpolated_img)
        if np.sum(interpolated_array) > 0:
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


### Center, and Crop to BOX NET
output_folder_image = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/image_crop' + ''.join(str(spacing))
output_folder_label = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/label_crop' + ''.join(str(spacing))
try:
    os.mkdir(output_folder_image)
    os.mkdir(output_folder_label)
except:
    print("directory already exists")

path_input = INTERP_DIR 
#PATH0='/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd'
dataset = 'E3311'
crop_shape = (118,118,32) # Shape of BoxNet
return_type='numpy_array'
for file in sorted(os.listdir(path_input)):
    if 'ct' not in file:
        label_id = '_'.join(file.split('_')[:2])
        patient_id = label_id.split('_')[0]
        print('label',label_id, 'patient', patient_id)
        path_to_label_nrrd = os.path.join(path_input,file)
        path_to_image_nrrd = os.path.join(path_input,dataset + '_' + patient_id + '_ct_interpolated_raw_raw_xx.nrrd')
        #output_folder_image = os.path.join(PATH0,'image_crop')
        #output_folder_label = os.path.join(PATH0,'label_crop')
        image_obj, label_obj = crop_roi_ene(dataset,
                                        label_id,
                                        label_id,
                                        path_to_image_nrrd, 
                                        path_to_label_nrrd, 
                                        crop_shape,
                                        return_type, 
                                        output_folder_image, 
                                        output_folder_label)