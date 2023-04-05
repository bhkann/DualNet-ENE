### Test Script for DualNet-ENE Model to Rune Inference ###
### Must be preceded by running "preprocessingroi_nrrd.py" script ###
### User must define path to saved model (see google drive link) ###

import os
import sys
from scipy import ndimage
import numpy as np
import pandas as pd
import SimpleITK as sitk

import tensorflow as tf
from keras.models import load_model

import tensorflow_addons as tfa
import time

tf.keras.backend.set_image_data_format('channels_last')

from model_callbacks import model_callbacks, roc_callback
from model_multi_dual import get_multi_dualnet, get_multi_smallnet, get_multi_boxnet
from sklearn.metrics import f1_score, confusion_matrix, precision_score, accuracy_score, hamming_loss, classification_report, roc_auc_score, matthews_corrcoef

from data_ene_test import get_data # MODIFY WITH DATA SCRIPT USED

#User-defined variables:
MODEL_PATH = '{PATH TO SAVED MODEL}'

# Hard-coded Hyperparameters for original model
classifier = 'multi'
OUTPUT_ACTIVATOR = 'sigmoid'
IMG_SIZE_PX = 118
SLICE_COUNT = 32
## Normalization parameters (must match model training parameters)
MIN = -175 
MAX = 275
n_classes = 2 # num of classifications / label classes
IMAGE_TYPE = 'ct'
IMAGE_SHAPE = (32, 118, 118) #z=128 or 96,x,y
INPUT_SHAPE = tuple([1] + list(IMAGE_SHAPE))
N_GPUS = 1 #CHANGE BASED ON SINGLE/MULTI GPU

## Select GPU for testing ##
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NAME = MODEL_NAME # 

# get data
data = get_data("test", IMAGE_SHAPE, image_format=IMAGE_TYPE, SAVE_CSV=False, min=MIN, max=MAX)


# load model
model = MODEL_PATH
original_model = load_model(model)
print('model name: ', model)

## Run predictions
no_results = []
label_predictions = []
labels = []
ids = []
index = 0
for patient in data:
    #### VARIABLES
    node_id = patient["node_id"]
    dataset = patient["dataset"]
    image = patient["arr_box"]
    image_small = patient["arr_small"]

    ## Final preprocessing step for original model, adds noise to cropped out voxels
    image = image + 0.001
    image[image > 1] = 1
    image[image < 0] = 0
    image_small = image_small + 0.001
    image_small[image_small > 1] = 1
    image_small[image_small < 0] = 0
    
    ## MODEL PREDICTION ##
    label_prediction = original_model.predict([image.reshape(1, *image.shape),image_small.reshape(1, *image_small.shape)],1)
    label_predictions.append(label_prediction)
    ids.append(node_id)
    index+=1

ids = np.array(ids)

label_predictions = np.array(label_predictions)
label_predictions = label_predictions.reshape(label_predictions.shape[0],label_predictions.shape[2])
results = np.column_stack((ids,label_predictions[:,0],label_predictions[:,1]))
results = pd.DataFrame(data=results, columns=["LN_ID","predict_pos","predict_ene"])

print(results)

SAVE_CSV = False
if SAVE_CSV:
    results.to_csv(os.path.join('~/DualNet-ENE_testresults.csv') # Can change filename as needed

