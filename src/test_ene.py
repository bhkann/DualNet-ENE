import os
import sys
from scipy import ndimage
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pprint import pprint
#
import tensorflow as tf

from keras.models import load_model

import tensorflow_addons as tfa
import time

tf.keras.backend.set_image_data_format('channels_last')

from model_callbacks import model_callbacks, roc_callback
from model_multi_dual import get_multi_dualnet, get_multi_smallnet, get_multi_boxnet
from sklearn.metrics import f1_score, confusion_matrix, precision_score, accuracy_score, hamming_loss, classification_report, roc_auc_score, matthews_corrcoef


from data_ene_test import get_data # MODIFY WITH UPDATED DATA SCRIPT

#LOAD LABELS
labels_df = pd.read_csv('LN_Label.csv')
labels_df['LN_ID'] = labels_df['LN_ID'].str.slice_replace(5,6,'-')

# Hyperparameters
classifier = 'multi'
OUTPUT_ACTIVATOR = 'sigmoid'

IMG_SIZE_PX = 118
SLICE_COUNT = 32
## Normalization parameters (must match model training parameters)
MIN = -175 #-175 #-175 #-175 WINDOW=450
MAX = 275 #275 #275 #275

CONTOUR_VAR = False # Test with random contour variation?
n_classes = 2 # num of classifications / label classes

IMAGE_TYPE = 'ct'
IMAGE_SHAPE = (32, 118, 118) #z=128 or 96,x,y
INPUT_SHAPE = tuple([1] + list(IMAGE_SHAPE))
BATCH_SIZE_PER_GPU = 6
N_GPUS = 1 #CHANGE BASED ON SINGLE/MULTI GPU

## Select GPU for testing ##
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MULTIGPU = False #CHANGE BASED ON SINGLE/MULTI GPU
BATCH_SIZE = 16 #BATCH_SIZE_PER_GPU * N_GPUS

NAME = MODEL_NAME # 

# create folder
dir_name = "HN_DL/OUTPUT/{}".format(NAME)
print("dir name: ", dir_name)
SAVE_DIR = "/media/bhkann/HN_RES1/" + dir_name  

# get data
data = get_data("test", IMAGE_SHAPE, MULTIGPU, image_format=IMAGE_TYPE, SAVE_CSV=False, SAVE_DIR=SAVE_DIR, dataset=DATASET, min=MIN, max=MAX, contour_var=CONTOUR_VAR)
    
label_predictions = []
no_results = []
MODEL_TO_USE = NAME[:1] #+ '1_final' #NAME # 'bestauc' # "_final" # "" or "_final"
SAVE_CSV = True

# load model
model = SAVE_DIR + "/" + MODEL_TO_USE + ".h5"
original_model = load_model(model)
print('model name: ', model)

SAVE_CASES = False

## Run predictions
label_predictions = []
labels = []
ids = []
index = 0
for patient in data:
    #### VARIABLES
    node_id = patient["node_id"]
    dataset = patient["dataset"]
    # formatted (cropped & reshaped) if MULTI_PREDICTION = False
    # not cropped or reshaped if MULTI_PREDICTION = True
    image = patient["arr_box"]
    image_small = patient["arr_small"]
    # original size
    #image_sitk_obj = patient["image_sitk_obj"]
    if 'size' in NAME:
        image_vol = np.sum(np.where(image>0,1,0))
        image_vol_norm = (image_vol - 1) / (20286 - 1) # FROM YST VALIDATION SET
        if image_vol_norm > 1:
            image_vol_norm = 1
        image_vol_norm = np.array(image_vol_norm, 'float32').reshape(1,)
        #print(image_vol, image_vol_norm, image_vol_norm.shape)
    image = image + 0.001
    image[image > 1] = 1
    image[image < 0] = 0
    image_small = image_small + 0.001
    image_small[image_small > 1] = 1
    image_small[image_small < 0] = 0
    
    label = patient["label"]
    
    if classifier == 'ene':
        label = label[1]
    elif classifier == 'pos':
        label = label[0]
    
    if SAVE_CASES == True:
        TEST_DIR = os.path.join(SAVE_DIR,'test_cases')
        try:
            os.mkdir(TEST_DIR)
        except:
            print("directory exists")
        #spacing = (1,1,3)
        sitk_object = sitk.GetImageFromArray(image)
        sitk_object_small = sitk.GetImageFromArray(image_small)
        #sitk_object.SetSpacing(spacing)
        #sitk_object.SetOrigin((0,0,0))
        #sitk_object.SetSpacing(spacing)
        writer = sitk.ImageFileWriter()
        writer.SetUseCompression(True)
        writer.SetFileName(os.path.join(TEST_DIR, "{}_{}_testcase.nrrd".format(dataset, node_id)))
        writer.Execute(sitk_object)
        writer.SetFileName(os.path.join(TEST_DIR, "{}_{}_small_testcase.nrrd".format(dataset, node_id)))
        writer.Execute(sitk_object_small)
        print("test case nrrd saved: ", node_id)
    if CONTOUR_VAR == True:
        TEST_DIR = os.path.join(SAVE_DIR,'ContourVariance_Studies')
        if index > 0 and index < 15 :
            sitk_object = sitk.GetImageFromArray(image)
            #sitk_object_small = sitk.GetImageFromArray(image_small)
            writer = sitk.ImageFileWriter()
            writer.SetUseCompression(True)
            writer.SetFileName(os.path.join(TEST_DIR, "{}_{}_testcase_var.nrrd".format(dataset, node_id)))
            writer.Execute(sitk_object)
            #writer.SetFileName(os.path.join(TEST_DIR, "{}_{}_small_testcase_var.nrrd".format(dataset, node_id)))
            #writer.Execute(sitk_object_small)
    #spacing = get_spacing(image_sitk_obj)
    if 'size' in NAME:
        if classifier == 'small':
            label_prediction = original_model.predict([image_small.reshape(1, *image_small.shape), image_vol_norm],1)
        else:
            label_prediction = original_model.predict([image.reshape(1, *image.shape),image_small.reshape(1, *image_small.shape), image_vol_norm],1)
    elif 'resnet' in NAME:
        label_prediction = original_model.predict([image.reshape(1, *image.shape)],1)
    else:
        label_prediction = original_model.predict([image.reshape(1, *image.shape),image_small.reshape(1, *image_small.shape)],1)
    label_predictions.append(label_prediction)
    labels.append(label)
    ids.append(node_id)
    index+=1

label_predictions = np.array(label_predictions)
#label_predictions = label_predictions.reshape(label_predictions.shape[0])
labels = np.array(labels)

ids = np.array(ids)
#ids = ids.reshape(ids.shape[0])


label_predictions = label_predictions.reshape(label_predictions.shape[0],label_predictions.shape[2])
results = np.column_stack((ids,labels,label_predictions[:,0],label_predictions[:,1]))
results_certain = pd.DataFrame(data=results, columns=["LN_ID","label_pos","label_ene","predict_pos","predict_ene"])

print(results_certain)

#Compute AUC
roc_test_ece = roc_auc_score(results_certain.label_ene, results_certain.predict_ene)
print('\rroc-auc_TEST:', str(round(roc_test_ece,4))+'\n')

threshold = np.arange(0.05,0.90,0.05)
acc = []
accuracies = []
YI = []
Youdens = []
best_threshold = np.zeros(len(results_certain))
y_prob = np.array(results_certain.predict_ene).astype('float') #change based on class
for j in threshold:
    y_pred = [1 if prob>=j else 0 for prob in y_prob]
    cm = confusion_matrix(results_certain['label_ene'].astype('float'), y_pred)  ##True, predicted
    #FP = cm.sum(axis=0) - np.diag(cm)  
    #FN = cm.sum(axis=1) - np.diag(cm)
    #TP = np.diag(cm)
    #TN = cm.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    #TPR = TP/(TP+FN)
    # Specificity or true negative rate
    #TNR = TN/(TN+FP) 
    TPR = cm[0,0]/(cm[0,0]+cm[0,1])
    TNR = cm[1,1]/(cm[1,0]+cm[1,1])
    Youden = TPR + TNR - 1
    Youdens.append(Youden)  #change metric based on class using

Youdens = np.array(Youdens)
Youdens[np.isnan(Youdens)] = 0
index = np.where(Youdens==Youdens.max()) 

YI.append(Youdens.max()) 
try:
    best_threshold = threshold[index[0][0]]
    print("best threshold YI: ", best_threshold)
except: 
    print('best threshold unable to be calculated, using 0.5')
    best_threshold = 0.5

predictecebin = np.copy(y_prob)
predictecebin[predictecebin >= best_threshold] = 1
predictecebin[predictecebin < best_threshold] = 0
print(classification_report(results_certain['label_ene'].astype('float'), predictecebin))
print('accuracy: ', accuracy_score(results_certain['label_ene'].astype('float'), predictecebin))

roc_test_ece = roc_auc_score(results_certain.label_ene, results_certain.predict_ene)
print('\rENE AUC certain:', str(round(roc_test_ece,6))+'\n')

roc_test_pos = roc_auc_score(results_certain.label_pos, results_certain.predict_pos)
print('\rPOS AUC certain:', str(round(roc_test_pos,6))+'\n')

# populate df
if SAVE_CSV:
    if CONTOUR_VAR == True:
        results_certain.to_csv(os.path.join(SAVE_DIR, 'ContourVariance_Studies', "testresults_contvar_{}_{}_{}.csv".format(DATASET,NAME,time.strftime("%Y%m%d-%H%M"))))
    else:
        results_certain.to_csv(os.path.join(SAVE_DIR, "testresults_{}_{}.csv".format(DATASET,NAME))) # Can choose lab drive vs HPC to save
        results_certain.to_csv(os.path.join('/media/bhkann/HN_RES1/HN_DL/E3311_DL/E3311_analysis', "testresults_{}_{}.csv".format(DATASET,NAME)))
