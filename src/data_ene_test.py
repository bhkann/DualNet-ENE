import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import filecmp
import sys
import pickle
#import h5py
import joblib
import random
from random import randint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from scipy.ndimage import interpolation
sys.path.append('/home/bhkann/git-repositories/hn-petct-net/data-utils')
from util import bbox2_3D
from rescale import rescale, downsample_img
import time
from scipy import ndimage
import cv2

## Read in data - NEED TO FUNCTIONALIZE THIS
#os.chdir("/home/bhkann/deeplearning/input/") #MODIFY WITH LOCATION OF DATA 
#os.chdir('/media/bhkann/HN_RES1/HN_DL/ENE_preprocess_nrrd/') #MODIFY WITH LOCATION OF DATA 
######## #WORKS WELL

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

def _contour_variance(arr):
    """
    Input Mask
    Returns image with mask variance, mimicking human variation
    """
    #mask3d = np.array(arr)
    #mask3d[mask3d>0]=1
    #listBoundary = []
    mask3d_NEW = np.array(arr)
    ### GLOBAL EROSION ###
    rand_e = randint(1,10)
    rand_d = randint(1,10)

    mask3d_err = ndimage.binary_erosion(arr, np.ones((1,rand_e,rand_e)), iterations=1)
    #print(np.sum(mask3d_err))

    ## FIND BOUNDING BOX ##
    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(mask3d_NEW) #was mask3d_err
    for r in range(rmin, rmax+1):
        rNow = mask3d_NEW[r]
        rNow = np.float32(rNow)
        _, threshold = cv2.threshold(rNow, 0, 1, cv2.THRESH_BINARY)
        threshold = threshold.astype(np.uint8)
        contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
        listContour = list(contours[0])    
        #print("NEW LIST:" , listContour)
        for c in listContour:  # contours[0]:
            #print(c)
            rand_01 = randint(0,5)
            R_ = r
            G_ = c[0][1]
            Y_ = c[0][0]
            #print([R_, G_, Y_])
            mask3d_NEW[R_, G_-rand_01:G_+rand_01, Y_-rand_01:Y_+rand_01] = 1
    ## Serial erosion, dilation, closing to create smoothing effect ##
    mask3d_NEW = ndimage.binary_erosion(mask3d_NEW, np.ones((1,2+rand_e,2+rand_e)), iterations=1)
    mask3d_NEW = ndimage.binary_dilation(mask3d_NEW, np.ones((1,2+rand_d,2+rand_d)), iterations=1)
    mask3d_NEW = ndimage.binary_closing(mask3d_NEW, np.ones((1,2,2)), iterations=2)
    #arr_new = np.multiply(mask3d_NEW, arr)

    return mask3d_NEW 

def generate_test_data(data_split, mode, model_input_size, image_format, min, max, contour_var):
    """
    Used for training and tuning only.
    data_split: dictionary of train, tune, and test split.
    mode: train, tune, or test
    """
    VERSION = "path"
    images = []
    images_small = []
    labels = []
    ids = []
    test = []
    MIN = min #-175
    MAX = max #275
    for idx, node in data_split[mode].iterrows(): #FIRST FOR TRAIN, THEN FOR TUNE
        dataset = node["dataset"]
        node_id = node["image_roi"]
        #node_cat = node["LNcat"]
        print("Path: ", node["image_"+VERSION])
        # get arr
        try:
            image_sitk_obj = sitk.ReadImage(node["image_"+VERSION])
            arr_image = sitk.GetArrayFromImage(image_sitk_obj) #get_arr(node["image_"+VERSION], mode, model_input_size)
            arr_image = np.clip(arr_image, a_min=MIN, a_max=MAX)
            #arr_image = arr_image - int(np.mean(arr_image))
            MAX, MIN = arr_image.max(), arr_image.min()
            arr_image = (arr_image - MIN) / (MAX - MIN)
            #arr_image = np.interp(arr_image,[-400,400],[0,1]) # ?-175 , 275 or -1024,3071
            arr_mask = get_arr(node["mask_"+VERSION], mode, model_input_size)
            print("image shape:", arr_image.shape, " mask shape: ", arr_mask.shape, "Max of label: ", arr_mask.max(), "min of label: ", arr_mask.min(), " Sum of label: ", np.sum(arr_mask))

            arr_box = np.multiply(arr_image,arr_mask)
            assertions(arr_image, arr_mask, dataset, node_id)
            #arr_box = arr_box + 0.001
            ## Generate SmallNet 32x32x32 volume
            resizedimension = 32
            #inputScaled = np.zeros((resizedimension,resizedimension,resizedimension), dtype='float32')
            rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(arr_box) ## Returns non-zero bounding box
            cropped_mask = arr_mask[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1]
            cropped_image = arr_image[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1]
            
            print('cropped_volume shape:', cropped_image.shape)
            oldspacing = list(cropped_image.shape)
            sizefactor = [resizedimension/x for x in oldspacing]
            resized_image = interpolation.zoom(cropped_image, zoom=sizefactor, mode='constant', order=1)
            resized_mask = interpolation.zoom(cropped_mask, zoom=sizefactor, mode='constant', order=0)
            #resized_mask = np.round(resized_mask)
            arr_small = np.multiply(resized_image,resized_mask)
            
            print("small net array size:", arr_small.shape)
            
            label=np.array([node['Positive'],node['ECE']])
            '''
            if contour_var == True:
                arr_mask = _contour_variance(arr_mask) ## Defaults: 3,6,3; 2D_var, erode_end, dilate_end
                print("Processed with contour variance")
                arr_box = np.multiply(arr_image,arr_mask)
            '''
            arr_box = arr_box.reshape(*arr_box.shape,1)
            arr_small = arr_small.reshape(*arr_small.shape,1)

            if contour_var == True:
                arr_box = arr_box + np.random.normal(0, 0.004, arr_box.shape)
                arr_small = arr_small + np.random.normal(0, 0.004, arr_small.shape)

            test.append(
                {"node_id": node_id,
                "dataset": dataset,
                "arr_box": arr_box,
                "arr_small": arr_small,
                "label": label})
            #images.append(arr_box)
            #images_small.append(arr_small)
            #labels.append(label)
            print ("{}_{}_{}".format(idx, dataset, node_id))
        except Exception as e: print(e) #("something went wrong")
    print("-------------")
    print("Pixels normalized to: Min: ", MIN, " Max: ", MAX)
    #return {"images": images_np, "labels": labels_np}
    return test


def get_data(mode, model_input_size, MULTI_PREDICTION, image_format, SAVE_CSV, SAVE_DIR, dataset, min, max, contour_var):
    
    SPACING = (1,1,3) #"" 
    print("SPACING: ", SPACING)
    ###### READ IN LABELING DATA (HN_LN_Label.xlsx) #####
    labels_df = pd.read_excel('/media/bhkann/HN_RES1/HN_DL/E3311_DL/E3311_LN_Label.xls') #, index_col=0)
    labels_df['LN_ID'] = labels_df['LN_ID'].str.slice_replace(5,6,'-')
    #labels_df.to_csv('/media/bhkann/HN_RES1/HN_DL/E3311_LN_LABEL.csv')
    #labels_df.LN_ID = labels_df['LN_ID'].str.replace('_','')
    #labels_df.LN_ID = pd.to_numeric(labels_df.LN_ID)

    ece_dict = {}
    ece_dict = dict(zip(labels_df.LN_ID, labels_df.ECE))
        
    positive_dict = {}
    positive_dict = dict(zip(labels_df.LN_ID, labels_df.Positive))

    lncat_dict = {}
    lncat_dict = dict(zip(labels_df.LN_ID, labels_df.LNcat)) #0,1,2

    certain_dict = {}
    certain_dict = dict(zip(labels_df.LN_ID, labels_df.CertaintyOverall)) #1,2,3

    PATH_image = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/image_crop' + str(SPACING)
    PATH_mask = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/label_crop' + str(SPACING)

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
    
    ## Combine labels and meta-data into spreadsheet ##
    labels_df = labels_df.rename(columns={'LN_ID':'image_roi'})
    
    df_combined = pd.merge(df,labels_df,how='left',on='image_roi')
    #df_combined['patient_id']=df_combined.image_roi.str[:-2]
    #df_combined.patient_id = pd.to_numeric(df_combined.patient_id)
    #df_combined = pd.merge(df_combined, ptdata_df,how='left',on='patient_id')

    df_combined['topcoder_id'] = range(1, len(df)+1)
    #random.seed(13)
    df_combined['topcoder_split'] = 'test'
    df_combined.groupby('topcoder_split')['ECE'].sum()
    df_combined.groupby('topcoder_split').count()
    
    if SAVE_CSV==True:
        #df_combined.to_csv('/home/bhkann/git-repositories/working-hn-dl-folder/nrrd_model/' + dataset + '_ene_input_' + time.strftime("%Y%m%d-%H%M") + '.csv')
        df_combined.to_csv(SAVE_DIR + '/' + dataset + '_ene_input_' + time.strftime("%Y%m%d-%H%M") + '.csv')
        print('input_data.csv saved to git repo')
    print('number of cases total:', len(df_combined))
    
    test = df_combined[df_combined["topcoder_split"]=="test"]
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

