### MULTILABEL CLASSIFIER ::: Combined BOX and SMALL ENE Deep Learning Convolutional Neural Networks Code for H/N ENE Deep Learning Project ###
### Code Written by Benjamin H. Kann (c), created 1/18/2018 ###

### Streamlined 1/18/2018 ###

#python3

import tensorflow as tf
import numpy as np
import random
from random import randint
from typing import List, Tuple
from tensorflow import keras
print('tensorflow version: ', tf.__version__)
print('keras version: ', keras.__version__)
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, merge, Conv3D, MaxPooling3D, UpSampling3D, LeakyReLU, PReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.layers.merge import concatenate
from keras.models import Model, load_model, model_from_json, Sequential
from keras.losses import BinaryCrossentropy
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras import regularizers, initializers
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler, EarlyStopping, History, CSVLogger, ReduceLROnPlateau
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.wrappers.scikit_learn import KerasClassifier
import h5py
import math
#K.set_image_data_format('channels_last')
#from sklearn.externals import joblib
#from scipy.ndimage.interpolation import map_coordinates
#from scipy.ndimage.filters import gaussian_filter
#from scipy.ndimage import binary_erosion, binary_closing, watershed_ift
#from scipy.ndimage.morphology import generate_binary_structure
#from scipy.ndimage.interpolation import rotate
#from sklearn.cluster import KMeans
#from sklearn import datasets, linear_model
#from sklearn.model_selection import train_test_split, StratifiedKFold, ShuffleSplit
#from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.datasets import make_classification
#from collections import Counter
#import imblearn
#from imblearn.over_sampling import SMOTE, RandomOverSampler
#import shutil

from sklearn.metrics import f1_score, confusion_matrix, precision_score, accuracy_score, hamming_loss, classification_report, roc_auc_score, matthews_corrcoef

#from keras.backend.tensorflow_backend import set_session

#tf.logging.set_verbosity(tf.logging.INFO)
#K.set_image_dim_ordering("tf")

# Set random number generator
#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# set_session(tf.Session(config=config))

###Configure TF GPU memory allocation
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config = config)

#############################################################################################################################
#keras.backend.clear_session()

#####DEFINE THE MODELS#####
# MULTI BOXNET ################################# BOXNET (32 x 118 x 118) ###### ECE+/ECE- BOUNDING BOX MODEL #############
### Data Characteristics
IMG_SIZE_PX = 118
SLICE_COUNT = 32
#n_classes = 2 # num of classifications / label classes

### Model Hyperparameters
#IMG_SIZE_PX = 118
#SLICE_COUNT = 32

####
#bLEARN_RATE = 0.001

###Dropout Config###
bdropout = 0.4 # was 0.4
bdropoutlast = 0.4 # was 0.4

sdropout = 0.5 # was 0.5
sdropoutlast = 0.5 # was 0.5

USE_DROPOUT = True


###L2 regularization (beta)
l2reg = 0.001 # was 0.001

###Type of classifier ('pos' or 'ece')
#classifier = 'ece'


def get_multi_dualnet(input_shape=(32, 118, 118, 1), load_weight_path=None, smallnet=True, metadata=False, metadata_shape=None, n_classes=2, output_activator='sigmoid', TRAINABLE_LAYERS=True) -> Model:
    
    input = Input(shape=input_shape, name='BoundingBoxInput')
    x = input
    #1st layer group
    x = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding="same")(x)  #to initialize weights: kernel_initializer='random_uniform', bias_initializer='zeros'
    x = Conv3D(64, (3, 3, 3), trainable=TRAINABLE_LAYERS, activation=None, padding='same', name='conv1', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization(name='bn1')(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)
    
    # 2nd layer group
    x = Conv3D(128, (3, 3, 3), trainable=TRAINABLE_LAYERS, activation=None, padding='same', name='conv2', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization(name='bn2')(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
    if USE_DROPOUT:
        x = Dropout(rate=bdropout)(x)
    
    # 3rd layer group
    x = Conv3D(256, (3, 3, 3), trainable=TRAINABLE_LAYERS, activation=None, padding='same', name='conv3a', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization(name='bn3a')(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3a')(x)
    x = Conv3D(256, (3, 3, 3), trainable=TRAINABLE_LAYERS, activation=None, padding='same', name='conv3b', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization(name='bn3b')(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3b')(x)
    if USE_DROPOUT:
        x = Dropout(rate=bdropout)(x)
    
    # 4th layer group
    x = Conv3D(512, (3, 3, 3), trainable=TRAINABLE_LAYERS, activation=None, padding='same', name='conv4a', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization(name='bn4a')(x)
    x = LeakyReLU(alpha=0.03)(x)
    #x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3a')(x)
    x = Conv3D(512, (3, 3, 3), trainable=TRAINABLE_LAYERS, activation=None, padding='same', name='conv4b', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization(name='bn4b')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = ZeroPadding3D(padding=(0,1,1))(x)
    #x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)
    if USE_DROPOUT:
        x = Dropout(rate=bdropout)(x)
    
    out_class = Flatten(name="out_class")(x)    
    out_class = Dense(2048, trainable=TRAINABLE_LAYERS, activation=None, kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(out_class) #was 2048
    out_class = BatchNormalization(name='bn5')(out_class)
    out_class = LeakyReLU(alpha=0.03)(out_class)
    out_class = Dropout(rate=bdropoutlast)(out_class)
    
    out_class = Dense(2048, trainable=True, activation=None, kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(out_class) # was 2048
    out_class = BatchNormalization(name='bn6')(out_class)
    out_class = LeakyReLU(alpha=0.03)(out_class)
    out_class = Dropout(rate=bdropoutlast)(out_class)
    
    if smallnet == True:
        input_smallnet = Input(shape=(32,32,32,1), name='input_s')
        s = input_smallnet
        #1st layer group
        s = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding="same")(s)  #to initialize weights: kernel_initializer='random_uniform', bias_initializer='zeros'
        s = Conv3D(64, (3, 3, 3), trainable=TRAINABLE_LAYERS, activation=None, padding='same', name='conv1s', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(s)
        s = BatchNormalization(name='bn1s')(s)
        s = LeakyReLU(alpha=0.03)(s)
        s = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1s')(s)
    
        # 2nd layer group
        s = Conv3D(128, (3, 3, 3), trainable=TRAINABLE_LAYERS, activation=None, padding='same', name='conv2s', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(s)
        s = BatchNormalization(name='bn2s')(s)
        s = LeakyReLU(alpha=0.03)(s)
        s = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2s')(s)
        if USE_DROPOUT:
            s = Dropout(rate=sdropout)(s)
    
        # 3rd layer group
        s = Conv3D(256, (3, 3, 3), trainable=TRAINABLE_LAYERS, activation=None, padding='same', name='conv3as', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(s)
        s = BatchNormalization(name='bn3as')(s)
        s = LeakyReLU(alpha=0.03)(s)
        s = Conv3D(256, (3, 3, 3), trainable=TRAINABLE_LAYERS, activation=None, padding='same', name='conv3bs', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(s)
        s = BatchNormalization(name='bn3bs')(s)
        s = LeakyReLU(alpha=0.03)(s)
        s = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3s')(s)
        if USE_DROPOUT:
            s = Dropout(rate=sdropout)(s)
    
        # 4th layer group
        s = Conv3D(512, (3, 3, 3), trainable=TRAINABLE_LAYERS, activation=None, padding='same', name='conv4as', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(s)
        s = BatchNormalization(name='bn4as')(s)
        s = LeakyReLU(alpha=0.03)(s)
        s = Conv3D(512, (3, 3, 3), trainable=TRAINABLE_LAYERS, activation=None, padding='same', name='conv4bs', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(s)
        s = BatchNormalization(name='bn4bs')(s)
        s = LeakyReLU(alpha=0.03)(s)
        #x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(x)
        s = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4s')(s)
        if USE_DROPOUT:
            s = Dropout(rate=sdropout)(s)
    
        s = Conv3D(64, (2, 2, 2), trainable=TRAINABLE_LAYERS, activation=None, name="last_64s", kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(s)
        s = BatchNormalization(name='bn_last_64s')(s)
        s = LeakyReLU(alpha=0.03)(s)
        s = Dropout(rate=sdropoutlast)(s)
        
        s = Conv3D(1, (1, 1, 1), trainable=True, activation=None, name="out_class_convs", kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(s)
        s = Flatten(name="out_small")(s)
        
        out_class = Dense(3, trainable=True, activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0))(out_class) # was 3
        #out_class = BatchNormalization()(out_class)
        #out_class = LeakyReLU(alpha=0.03)(out_class)
        
        out_small = concatenate([s, out_class])
        out_small = Dense(4, activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.0))(out_small) # was 4
        #out_small = BatchNormalization()(out_small)
        #out_small = LeakyReLU(alpha=0.03)(out_small)
        
    ### ADDING METADATA - SIZE, HPV status ###
    if metadata == True:
        input_metadata = Input(shape=metadata_shape)
        #x2 = Dense(16, activation=None)(input_metadata)
        #x2 = LeakyReLU(alpha=0.03)(x2)
        
        #out_small = Dense(9, activation=None)(out_small)
        #out_class = BatchNormalization()(out_class)
        out_small = LeakyReLU(alpha=0.03)(out_small)
        conc = concatenate([input_metadata, out_small]) # merge in metadata
        
        conc = Dense(4, activation=None, kernel_regularizer = regularizers.l2(0.0))(conc)
        #conc = BatchNormalization()(conc)
        conc = LeakyReLU(alpha=0.03)(conc)
        #conc = Dropout(rate=bdropoutlast)(conc)
        
    if smallnet == True and metadata == False:       
        out_smallbox = Dense(n_classes, activation=output_activator)(out_small)
        model = Model(inputs=[input, input_smallnet], outputs=[out_smallbox])
    elif smallnet == True and metadata == True:
        out_metasmall = Dense(n_classes, activation="sigmoid")(conc)
        model = Model(inputs=[input, input_smallnet, input_metadata], outputs=[out_metasmall])
    elif smallnet == False and metadata == True:
        model = Model(inputs=[input, input_metadata], outputs=[out_meta])
    elif smallnet == False and metadata == False:
        out_class = Dense(n_classes, activation="sigmoid")(out_class)
        model = Model(inputs=input, outputs=[out_class])
    
    ### Load pre-trained weights
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
        
    #model.compile(optimizer=SGD(lr=bLEARN_RATE, momentum=0.9, nesterov=True, decay=0), loss="binary_crossentropy", metrics=["accuracy"])
    #model.compile(optimizer=Adam(lr=bLEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss="binary_crossentropy", metrics=["AUC"])
    #model.compile(optimizer=RMSprop(lr=LEARN_RATE, rho=0.9, decay=0.0), loss="binary_crossentropy", metrics=["binary_accuracy"])
    #model.compile(optimizer=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004), loss="binary_crossentropy", metrics=["binary_accuracy"])
    #if features:
    #    model = Model(input=inputs, output=[last64])
    model.summary(line_length=140)
    #model.get_config()
    
    return model


#weightsload = 'model_dual001_kfold1'
## Save the BOX model - ece
#model = get_multi_boxnet(input_shape=(32, 118, 118, 1), smallnet=True, metadata=False, metadata_shape=(1,), load_weight_path=None) #'/home/ec2-user/src/data/experiments/' + weightsload + '_bestvalloss_weights.hdf5'

## Title the model / Identify
#modelname = 'model_dual001_01'

#CPU - Save model summary - with hyperparameters
#from sklearn.externals import joblib
'''
with open('/home/ec2-user/src/data/experiments/' + modelname + '_summary.txt','w') as fh:
    #AWS: '~/src/data/experiments/'
    #Macbook: '/Volumes/BK_RESEARCH/HN_DL/HN_DL_DATA/Code/experiments/' + modelname + '_summary.txt'
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
    ### Model Hyperparameters
    fh.write('Hyperparameters:' 'IMG_SIZE_PX=' + ', SLICE_COUNT=' + str(32) + '\n' + 'Classifier type:' + classifier +
        ' Number Classes:' + str(n_classes) + '\n' + 'Learning Rate: ' + str(bLEARN_RATE) + '\n' + 'Dropout: L2,3,4,Last:' +
        str(bdropout) + ',' + str(bdropoutlast) + '\n' + 'L2 regularization (beta): ' + str(l2reg) + '(all conv)') # + 'class weights:' + str(weights))

#model.summary(line_length=140)
#CPU - Save full model configuration in dictionary form (pickled)
modelconfig = model.get_config()
joblib.dump(modelconfig, '/home/ec2-user/src/data/experiments/' + modelname + '_configuration.pkl')
'''



#callbacks=[roc_callback(training_data=training_data,validation_data=validation_data)] 


## Function - Train the Model
def trainmodel(x_train, y_train, x_eval, y_eval, batch_size, epochs, cweight=None, cbacks=None):
    myhistory = model.fit(x_train, y_train,
        batch_size, epochs,
        verbose=1,
        validation_data=(x_eval, y_eval), 
        callbacks=cbacks,
        class_weight=cweight) #validation_data=(x_minieval, y_eval)
    return myhistory

    ## Use the data aug generator
'''def trainmodel_gen(steps_per_epoch, epochs):
    myhistory = model.fit_generator(train_gen, steps_per_epoch, epochs, validation_data=(x_eval, y_eval), verbose=1, callbacks=[bestcheckpoint, csvlog, learnrate_scheduler])
    return myhistory'''

### Test model
def testmodel(x, y, batch_size):
    score = model.evaluate(x, y, batch_size=batch_size, verbose=0)
    predict = model.predict(x, batch_size=batch_size) #None batch defaults to 32
    print(predict)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return score, predict

############### TRAINING ECE SMALLNET MODEL #############

# 2) SMALLNET
### Load SMALLNET Data ###

def get_multi_smallnet(input_shape=(32, 32, 32, 1), load_weight_path=None, metadata=None, metadata_shape=None, n_classes=2) -> Model:
    
    input = Input(shape=input_shape, name="input_s")
    x = input
    #1st layer group
    x = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding="same")(x)  #to initialize weights: kernel_initializer='random_uniform', bias_initializer='zeros'
    x = Conv3D(64, (3, 3, 3), activation=None, padding='same', name='conv1', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)
    
    # 2nd layer group
    x = Conv3D(128, (3, 3, 3), activation=None, padding='same', name='conv2', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
    if USE_DROPOUT:
        x = Dropout(rate=sdropout)(x)
    
    # 3rd layer group
    x = Conv3D(256, (3, 3, 3), activation=None, padding='same', name='conv3a', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = Conv3D(256, (3, 3, 3), activation=None, padding='same', name='conv3b', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)
    if USE_DROPOUT:
        x = Dropout(rate=sdropout)(x)
    
    # 4th layer group
    x = Conv3D(512, (3, 3, 3), activation=None, padding='same', name='conv4a', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = Conv3D(512, (3, 3, 3), activation=None, padding='same', name='conv4b', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    #x = ZeroPadding3D(padding=(0,1,1))(x)
    #x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)
    if USE_DROPOUT:
        x = Dropout(rate=sdropout)(x)
    
    x = Conv3D(64, (2, 2, 2), activation=None, name="last_64", kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.03)(x)
    #out_class = Dropout(rate=sdropoutlast)(last64)
    
    out_class = Conv3D(1, (1, 1, 1), activation=None, name="out_class_conv", kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    out_class = Flatten(name="out_class")(x)
    
    ### ADDING METADATA - SIZE, HPV status ###
    if metadata == True:
        input_metadata = Input(shape=metadata_shape)
        #x2 = Dense(25, activation= 'relu')(input_metadata) 
        #out_class = Dense(5, activation=None)(out_class)
        #out_class = LeakyReLU(alpha=0.03)(out_class)
        conc = concatenate([input_metadata,out_class]) # merge in metadata
        #conc = Dense(4, activation=None, kernel_regularizer = regularizers.l2(0.0))(conc)
        #conc = BatchNormalization()(conc)
        #conc = LeakyReLU(alpha=0.03)(conc)
        #conc = Dropout(rate=sdropoutlast)(conc)
        out_meta = Dense(n_classes, activation="sigmoid")(conc)
        
        model = Model(inputs=[input, input_metadata], outputs=[out_meta])
    else:
        out_class = Dense(n_classes, activation="sigmoid")(out_class)
        model = Model(inputs=input, outputs=[out_class])
    
    ### Load pre-trained weights
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
        
    #model.compile(optimizer=SGD(lr=sLEARN_RATE, momentum=0.9, nesterov=True, decay=0), loss="binary_crossentropy", metrics=["accuracy"])
    #model.compile(optimizer=Adam(lr=bLEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss="binary_crossentropy", metrics=["accuracy"])
    #model.compile(optimizer=RMSprop(lr=LEARN_RATE, rho=0.9, decay=0.0), loss="binary_crossentropy", metrics=["binary_accuracy"])
    #model.compile(optimizer=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004), loss="binary_crossentropy", metrics=["binary_accuracy"])
    #if features:
    #    model = Model(input=inputs, output=[last64])
    model.summary(line_length=140)
    #model.get_config()
    
    return model



def get_multi_boxnet(input_shape=(32, 118, 118, 1), load_weight_path=None, metadata=None, metadata_shape=None) -> Model:
    
    input = Input(shape=input_shape, name="input_b")
    x = input
    #1st layer group
    x = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding="same")(x)  #to initialize weights: kernel_initializer='random_uniform', bias_initializer='zeros'
    x = Conv3D(64, (3, 3, 3), activation=None, padding='same', name='conv1', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)
    
    # 2nd layer group
    x = Conv3D(128, (3, 3, 3), activation=None, padding='same', name='conv2', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
    if USE_DROPOUT:
        x = Dropout(rate=bdropout)(x)
    
    # 3rd layer group
    x = Conv3D(256, (3, 3, 3), activation=None, padding='same', name='conv3a', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)
    x = Conv3D(256, (3, 3, 3), activation=None, padding='same', name='conv3b', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)
    if USE_DROPOUT:
        x = Dropout(rate=bdropout)(x)
    
    # 4th layer group
    x = Conv3D(512, (3, 3, 3), activation=None, padding='same', name='conv4a', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = Conv3D(512, (3, 3, 3), activation=None, padding='same', name='conv4b', strides=(1, 1, 1), kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.03)(x)
    x = ZeroPadding3D(padding=(0,1,1))(x)
    #x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')(x)
    if USE_DROPOUT:
        x = Dropout(rate=bdropout)(x)
    
    #x = Conv3D(128, (2, 2, 2), activation=None, name="last_64", kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(x)
    #x = BatchNormalization()(x)
    #last64 = Activation('relu')(last64)
    #out_class = LeakyReLU(alpha=0.03)(x)
    #x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(x)
    #out_class = Dropout(rate=bdropoutlast)(out_class)
    #out_class = Conv3D(1, (1, 1, 1), activation=None, name="out_class_conv", kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(l2reg))(out_class)
    
    out_class = Flatten(name="out_class")(x)    
    out_class = Dense(2048, activation=None, kernel_regularizer = regularizers.l2(l2reg))(out_class)
    out_class = BatchNormalization()(out_class)
    out_class = LeakyReLU(alpha=0.03)(out_class)
    out_class = Dropout(rate=bdropoutlast)(out_class)
    
    out_class = Dense(2048, activation=None, kernel_regularizer = regularizers.l2(l2reg))(out_class)
    out_class = BatchNormalization()(out_class)
    out_class = LeakyReLU(alpha=0.03)(out_class)
    out_class = Dropout(rate=bdropoutlast)(out_class)
    
    #out_class = Dense(2048, activation=None, kernel_regularizer = regularizers.l2(l2reg))(out_class)
    #out_class = BatchNormalization()(out_class)
    #out_class = LeakyReLU(alpha=0.03)(out_class)
    #out_class = Dropout(rate=bdropoutlast)(out_class)
    
    ### ADDING METADATA - SIZE, HPV status ###
    if metadata == True:
        input_metadata = Input(shape=metadata_shape)
        x2 = Dense(16, activation=None)(input_metadata)
        x2 = LeakyReLU(alpha=0.03)(x2)
        
        out_class = Dense(48, activation=None)(out_class)
        #out_class = BatchNormalization()(out_class)
        out_class = LeakyReLU(alpha=0.03)(out_class)
        conc = concatenate([x2, out_class]) # merge in metadata
        
        conc = Dense(32, activation=None, kernel_regularizer = regularizers.l2(0.0))(conc)
        #conc = BatchNormalization()(conc)
        conc = LeakyReLU(alpha=0.03)(conc)
        #conc = Dropout(rate=bdropoutlast)(conc)
        out_meta = Dense(n_classes, activation="sigmoid")(conc)
        
        model = Model(inputs=[input, input_metadata], outputs=[out_meta])
    else:
        
        out_class = Dense(n_classes, activation="sigmoid")(out_class)
        model = Model(inputs=input, outputs=[out_class])
        
    ### Load pre-trained weights
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
        
    #model.compile(optimizer=SGD(lr=bLEARN_RATE, momentum=0.9, nesterov=True, decay=0), loss="binary_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=Adam(lr=bLEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss="binary_crossentropy", metrics=["accuracy"])
    #model.compile(optimizer=RMSprop(lr=LEARN_RATE, rho=0.9, decay=0.0), loss="binary_crossentropy", metrics=["binary_accuracy"])
    #model.compile(optimizer=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004), loss="binary_crossentropy", metrics=["binary_accuracy"])
    #if features:
    #    model = Model(input=inputs, output=[last64])
    model.summary(line_length=140)
    #model.get_config()
    
    return model