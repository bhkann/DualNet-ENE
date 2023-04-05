import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import filecmp
import sys
import pickle
import joblib
import random
from random import randint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from scipy.ndimage import interpolation
sys.path.append('/utils')
from util import bbox2_3D
from rescale import rescale, downsample_img
import time
from scipy import ndimage
import cv2

### This code (called from test_ene.py pulls cropped scans/masks, clips and normalizes pixel intensities, and generates SmallNet output for the DualNet)
### Helper Functions ###

def print_shape(obj, mode):
    print ("{} image shape :: {} \n{} label shape :: {}".format(
        mode, obj["images"].shape,
        mode, obj["labels"].shape))

def get_arr(path_to_nrrd, mode, model_input_size):
    """
    Reads a nrrd file and spits out a numpy array.
    path_to_nrrd: path_to_nrrd
    mode: train or tune
    model_input_size: tuple of model input
    """
    print("path to nrrd image:", path_to_nrrd)
    sitk_image = sitk.ReadImage(path_to_nrrd)
    arr = sitk.GetArrayFromImage(sitk_image)
    #if mode == "tune":
    #    arr = format_arr(arr, model_input_size)
    return arr

def crop_arr(arr, model_input_size):
    start_z = arr.shape[0]//2 - model_input_size[0]//2
    start_y = arr.shape[1]//2 - model_input_size[1]//2
    start_x = arr.shape[2]//2 - model_input_size[2]//2
    #
    arr = arr[start_z:start_z+model_input_size[0],
              start_y:start_y+model_input_size[1],
              start_x:start_x+model_input_size[2]]
    return arr

def format_arr(arr, model_input_size):
    """
    Used for test mode. Crops and reshapes array.
    Also remaps image values.
    """
    arr = crop_arr(arr, model_input_size)
    arr = arr.reshape(1, *arr.shape)
    return arr

def assertions(arr_image, arr_label, dataset, patient_id):
    assert arr_image.shape == arr_label.shape, "image and label do not have the same shape."
    assert arr_label.min() == 0, "label min is not 0 @ {}_{}".format(dataset, patient_id)
    assert arr_label.max() == 1, "label max is not 1 @ {}_{}".format(dataset, patient_id)
    assert len(np.unique(arr_label))==2, "length of label unique vals is not 2 @ {}_{}".format(dataset, patient_id)


def generate_test_data(data_split, mode, model_input_size, image_format, min, max):
    images = []
    images_small = []
    labels = []
    ids = []
    test = []
    MIN = min #-175
    MAX = max #275
    for idx, node in data_split[mode].iterrows(): 
        dataset = node["dataset"]
        node_id = node["image_roi"]
        print("Path: ", node["image_path"])
        try:
            image_sitk_obj = sitk.ReadImage(node["image_path"])
            arr_image = sitk.GetArrayFromImage(image_sitk_obj) 
            arr_image = np.clip(arr_image, a_min=MIN, a_max=MAX)
            MAX, MIN = arr_image.max(), arr_image.min()
            arr_image = (arr_image - MIN) / (MAX - MIN)
            arr_mask = get_arr(node["mask_path"], mode, model_input_size)
            print("image shape:", arr_image.shape, " mask shape: ", arr_mask.shape, "Max of label: ", arr_mask.max(), "min of label: ", arr_mask.min(), " Sum of label: ", np.sum(arr_mask))

            arr_box = np.multiply(arr_image,arr_mask)
            assertions(arr_image, arr_mask, dataset, node_id)

            ## Generate the SmallNet input
            resizedimension = 32
            rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(arr_box) ## Returns non-zero bounding box
            cropped_mask = arr_mask[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1]
            cropped_image = arr_image[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1]
            
            print('cropped_volume shape:', cropped_image.shape)
            oldspacing = list(cropped_image.shape)
            sizefactor = [resizedimension/x for x in oldspacing]
            resized_image = interpolation.zoom(cropped_image, zoom=sizefactor, mode='constant', order=1)
            resized_mask = interpolation.zoom(cropped_mask, zoom=sizefactor, mode='constant', order=0)
            arr_small = np.multiply(resized_image,resized_mask)
            
            print("small net array size:", arr_small.shape)
            
            arr_box = arr_box.reshape(*arr_box.shape,1)
            arr_small = arr_small.reshape(*arr_small.shape,1)

            test.append(
                {"node_id": node_id,
                "dataset": dataset,
                "arr_box": arr_box,
                "arr_small": arr_small})
            print ("{}_{}_{}".format(idx, dataset, node_id))
        except Exception as e: print(e)
    print("-------------")
    return test


def get_data(mode, model_input_size, MULTI_PREDICTION, image_format, SAVE_CSV, min, max):
    
    SPACING = (1,1,3) #"" 
    print("SPACING: ", SPACING)

    PATH_image = '~/DualNet-ENE/image_crop' + str(SPACING)
    PATH_mask = '~/DualNet-ENE/label_crop' + str(SPACING)

    master_list_image_path = [os.path.join(PATH_image,f) for f in sorted(os.listdir(PATH_image))]
    master_list_mask_path = [os.path.join(PATH_mask,f) for f in sorted(os.listdir(PATH_mask))]
    master_list_image = [f.split('/')[-1] for f in master_list_image_path]
    master_list_mask = [f.split('/')[-1] for f in master_list_mask_path]
    master_list_roi_image = ['_'.join(f.split('_')[1:3]) for f in master_list_image]
    master_list_roi_mask = ['_'.join(f.split('_')[1:3]) for f in master_list_mask]

    try:
        assert master_list_roi_image == master_list_roi_mask
    except: print('image and mask lists do not match!!!')
            
    print("images: ", len(master_list_roi_image), " masks: ", len(master_list_roi_mask))
    
    ### CREATE THE MASTER DATAFRAME (excludes unpaired images/labels)    
    master_list = list(zip(master_list_image_path,master_list_mask_path,master_list_image,master_list_mask,master_list_roi_image,master_list_roi_mask))
    df = pd.DataFrame(master_list, columns=['image_path','mask_path','image_file','mask_file','image_roi','mask_roi'])
    
    try:
        assert (df.image_roi.equals(df.mask_roi))
    except:
        print("Label and Image lists do not match!!!")
    
    df['dataset'] = df.image_file.str.split('_').str[0]
    df['patient_id'] = df.image_roi.str.split('_').str[0]

    df['topcoder_id'] = range(1, len(df)+1)
    #random.seed(13)
    df['topcoder_split'] = 'test'
    df.groupby('topcoder_split')['ECE'].sum()
    df.groupby('topcoder_split').count()
    
    if SAVE_CSV==True:
        #df_combined.to_csv('/home/bhkann/git-repositories/working-hn-dl-folder/nrrd_model/' + dataset + '_ene_input_' + time.strftime("%Y%m%d-%H%M") + '.csv')
        df.to_csv('~/DualNet-ENE/ene_input_' + time.strftime("%Y%m%d-%H%M") + '.csv')
        print('input_data.csv saved to git repo')
    print('number of cases total:', len(df))
    
    test = df[df["topcoder_split"]=="test"]
    data_split = {
        "test" : test}

    print("Length of test: ", len(test))
    ### PICK UP HERE WHERE LEFT OFF ###
    
    if mode=="train_tune":
        data = {
            "train": generate_train_tune_data(data_split, "train", model_input_size, image_format, min, max),
            "tune": generate_train_tune_data(data_split, "tune", model_input_size, image_format, min, max)
        }
        print_shape(data["train"], "train")
        print_shape(data["tune"], "tune")
    elif mode=="test":
        data = generate_test_data(data_split, "test", model_input_size, image_format, min, max, contour_var)
        print ("test cases :: {}\ntest image box shape :: {}".format(len(data), data[0]["arr_box"].shape))
    return data

