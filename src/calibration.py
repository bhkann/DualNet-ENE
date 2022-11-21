## CALIBRATION ##
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def ece_score(probabilities, accuracy, confidence):   
    
    n_bins = len(accuracy) 
    n = len(probabilities) 
    h = np.histogram(a=probabilities, range=(0, 1), bins=n_bins)[0]  
    
    ece = 0
    for m in np.arange(n_bins):
        ece = ece + (h[m] / n) * np.abs(accuracy[m] - confidence[m])
    return ece

### CALCULATE RELIABILITY DIAGRAM ON TEST ###
#results_df = pd.read_csv('/media/bhkann/HN_RES1/HN_DL/OUTPUT/1_ene_multi_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_-175_275_20211210-1356_notest/testresults_E3311_1_ene_multi_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_-175_275_20211210-1356_notest.csv')
DATASET='YST' #'E3311'
DIR = '/media/bhkann/HN_RES1/HN_DL/OUTPUT/'
FOLDER_NAME = "1_ene_multi_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_-175_275_CERTFalse_20220208-1531"
results_df = pd.read_csv(DIR + FOLDER_NAME + '/testresults_' + DATASET + '_' + FOLDER_NAME + '.csv') # + '_kfold_ensemble.csv')

#results_df = results_df.rename(columns={'Unnamed: 0':'LN_ID'})
#results_df = results_df.iloc[2:,:]
#results_df = results_df.reset_index()
#results_df = results_df[results_df['CertaintyECE']==1]
df_certain = results_df[results_df["CertaintyECE"] == 1]

y_test = np.array(df_certain['label_ene']).astype('float')
y_test = y_test.reshape(y_test.shape[0],1)
pred_test = np.array(df_certain.predict_ene).astype('float') #change based on class
pred_test = pred_test.reshape(pred_test.shape[0],1)

#for i in range(y_test.shape[0]):
#    acc_before[i], prob_before[i] = calibration_curve(y_prob=pred_test[:, i], y_true=y_test[:, i], 
#                                                      n_bins=10, normalize=True)
prob_true, prob_pred = calibration_curve(y_test, pred_test, n_bins=10, normalize=False)

### PLOT RELIABILITY DIAGRAM ###
import matplotlib.pyplot as plt
plt.figure(figsize=(16,5))

plt.subplot(121)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(prob_true, prob_pred, marker='.', label='before calibration')
plt.legend(); plt.title('ENE Calibration'); plt.ylabel('fraction of positive'); plt.xlabel('probabilities')

plt.show()

ece_score(pred_test, prob_true, prob_pred)


prob_true_cal, prob_pred_cal = calibration_curve(ground_truth, confidences, n_bins=10, normalize=False)
plt.subplot(122)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(prob_true_cal, prob_pred_cal, marker='.', label='after calibration')
plt.legend(); plt.title('BadBuy'); plt.ylabel('fraction of positive'); plt.xlabel('probabilities')
plt.show()
### ECE SCORE FOR GOOD BUY ###
ece_score(pred_test, prob_true, prob_pred)

ece_score(pred_test[:,0], acc_before[0], prob_before[0])

### ECE SCORE FOR BAD BUY ###
ece_score(pred_test[:,1], acc_before[1], prob_before[1])



### NetCal: https://github.com/fabiankueppers/calibration-framework
### virtualenv: calibration-ene
### CALIBRATE ON VALIDATION SET ###

import numpy as np
import pandas as pd
from netcal.scaling import TemperatureScaling, LogisticCalibration
from netcal.binning import HistogramBinning, BBQ, IsotonicRegression, ENIR
from netcal.presentation import ReliabilityDiagram

DATASET='YST' #'E3311'
DIR = '/media/bhkann/HN_RES1/HN_DL/OUTPUT/'
FOLDER_NAME = "1_ene_multi_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_-175_275_CERTFalse_20220208-1531"
#results_df = pd.read_csv(DIR + FOLDER_NAME + '/testresults_' + DATASET + '_' + FOLDER_NAME + '.csv') # + '_kfold_ensemble.csv')

y_val = np.array(df_certain['label_ene']).astype('float')
pred_val = np.array(df_certain.predict_ene).astype('float') #change based on class

results_e3311 = pd.read_csv(DIR + FOLDER_NAME + '/testresults_E3311_' + FOLDER_NAME + '.csv') # + '_kfold_ensemble.csv')
dfe3311_certain = results_e3311[results_e3311["CertaintyECE"] == 1]
y_ext = np.array(dfe3311_certain['label_ene']).astype('float')
pred_ext = np.array(dfe3311_certain.predict_ene).astype('float') #change based on class

temperature = LogisticCalibration()  # PLATT SCALING #
temperature.fit(pred_val, y_val)
calibrated = temperature.transform(pred_ext)

dfe3311_certain['calibrated_ene']=calibrated
dfe3311_certain[['predict_ene','calibrated_ene']]

dfe3311_certain.to_csv(DIR + FOLDER_NAME + '/testresults_E3311_' + FOLDER_NAME + '_calibrated.csv')
#histobin = ENIR() ## Works very well ENIR BEST?? not too perfect?
#histobin.fit(confidences, ground_truth)
#calibrated = histobin.transform(pred_ext)

n_bins = 10
diagram = ReliabilityDiagram(n_bins)
diagram.plot(pred_ext, y_ext).show()  # visualize miscalibration of uncalibrated
diagram.plot(calibrated, y_ext).show()   # visualize miscalibration of calibrated


from netcal.metrics import ECE, MCE

n_bins = 10

ece = ECE(n_bins)
uncalibrated_score = ece.measure(pred_ext,1-pred_ext)
calibrated_score = ece.measure(calibrated,1-calibrated)
print('uncalibrated: ', uncalibrated_score, ' calibrated: ', calibrated_score)




prob_true, prob_pred = calibration_curve(y_ext, pred_ext, n_bins=10, normalize=False)
plt.subplot(121)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(prob_true, prob_pred, marker='.', label='Before Calibration')
plt.legend(); plt.title('Before Calibration'); plt.ylabel('Fraction of Actual ENE'); plt.xlabel('Predicted Probability of ENE')
#plt.show()

prob_true_cal, prob_pred_cal = calibration_curve(y_ext, calibrated, n_bins=10, normalize=False)
plt.subplot(122)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(prob_true_cal, prob_pred_cal, marker='.', label='After Calibration')
plt.legend(); plt.title('After Calibration'); plt.ylabel('Fraction of Actual ENE'); plt.xlabel('Predicted Probability of ENE')

plt.savefig('/media/bhkann/HN_RES1/HN_DL/OUTPUT/1_ene_multi_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_-175_275_CERTFalse_20220208-1531/calibration_plot_e3311_paper.png', dpi=600, facecolor='w', edgecolor='w',
        orientation='portrait', format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
plt.show()


from sklearn.metrics import f1_score, confusion_matrix, precision_score, accuracy_score, hamming_loss, classification_report, roc_auc_score, matthews_corrcoef

roc_test_ece = roc_auc_score(y_ext, calibrated)
print('\rroc-auc_TEST certain macro=1 else =0:', str(round(roc_test_ece,4))+'\n')

roc_auc_score(y_ext, pred_ext)

print("CALIBRATED: Using Threshold 0.21")
predictecebin = np.copy(calibrated)
predictecebin[predictecebin >= 0.21] = 1
predictecebin[predictecebin < 0.21] = 0
print(classification_report(y_ext, predictecebin))
print('accuracy: ', accuracy_score(y_ext, predictecebin)) 


print("UNCALIBRATED: Using Threshold 0.3")
predictecebin = np.copy(pred_ext)
predictecebin[predictecebin >= 0.55] = 1
predictecebin[predictecebin < 0.55] = 0
print(classification_report(y_ext, predictecebin))
print('accuracy: ', accuracy_score(y_ext, predictecebin)) 

'''
## ENV env-hn-3dunet
## Then calibration-ene

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
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import tensorflow_addons as tfa

tf.keras.backend.set_image_data_format('channels_last')

sys.path.append('/home/bhkann/git-repositories/working-hn-dl-folder/nrrd_model')
from model_callbacks import model_callbacks, roc_callback
#from model_3d.model import isensee2017_model # FIXED (was error d/t InstanceNormalization in "model.py")
from model_multi_dual import get_multi_dualnet, get_multi_smallnet, get_multi_boxnet
from sklearn.metrics import f1_score, confusion_matrix, precision_score, accuracy_score, hamming_loss, classification_report, roc_auc_score, matthews_corrcoef

from data_ene_060321_YST import get_data 
data = get_data("tune", (32, 118, 118), False, image_format='ct', SAVE_CSV=False, SAVE_DIR='', dataset='YST', no_test='traintune', certain=False, min=-175, max=275) #seg target can be gtv, gtvn, gtvp


classifier = 'multi'
OUTPUT_ACTIVATOR = 'sigmoid'
# Hyperparameters
IMG_SIZE_PX = 118
SLICE_COUNT = 32
DATASET='YST'
## Normalization parameters (must match model training parameters)
MIN = -175 #-175 #-175 #-175
MAX = 275 #275 #275 #275
NAME = "1_ene_multi_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_-175_275_CERTFalse_20220208-1531" #1_ene_multi_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_-175_275_CERTFalse_20220208-1531" #1_ene_multi_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_-175_275_CERTFalse_E3311_FINETUNE_20220209-1612" #"1_ene_single_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_20210820-1037" #"1_ene_single_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_20210820-1037"
#BEST SO FAR: 1_ene_single_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_20210820-1037" #"1_ene_single_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_20210820-1037" # loss.__name__.replace("_", "-") + "-0.0005"
#MODEL = "1_ene_single_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_20210827-1505" #"/home/bhkann/deeplearning/output/1_wce-dice-loss-0.001-augment-hn-petct-pmhmdchuschum_gtvn/1.h5" #"/output/111_focal-tversky-loss-0.0005-augment-maastro/111.h5"

# create folder
dir_name = "HN_DL/OUTPUT/{}".format(NAME)
print("dir name: ", dir_name)
SAVE_DIR = "/media/bhkann/HN_RES1/" + dir_name  

label_predictions = []
no_results = []
MODEL_TO_USE = NAME[:1] #+ '1_final' #NAME # 'bestauc' # "_final" # "" or "_final"
SAVE_CSV = True

MULTI_PREDICTION = False
# load model
model = SAVE_DIR + "/" + MODEL_TO_USE + ".h5"
original_model = load_model(model)
print('model name: ', model)
## Combining separate GTVP and GTVN models ##
SAVE_CASES = False


## Run predictions
label_predictions = []
labels = []
ids = []
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
    image_small = image_small + 0.001
    image_small[image_small > 1] = 1
    
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

label_predictions = np.array(label_predictions)
#label_predictions = label_predictions.reshape(label_predictions.shape[0])
labels = np.array(labels)

ids = np.array(ids)
#ids = ids.reshape(ids.shape[0])

#print("label_predictions:", label_predictions)

if classifier == 'ene' or classifier == 'pos':
    labels = labels.reshape(labels.shape[0])
    if OUTPUT_ACTIVATOR == 'sigmoid':
        #label_predictions = label_predictions.reshape(label_predictions.shape[0])
        results = np.column_stack((ids,labels,label_predictions[:,0][:,0]))
    elif OUTPUT_ACTIVATOR == 'softmax':
        results = np.column_stack((ids,labels,label_predictions[:,0][:,1]))
    results_df = pd.DataFrame(data=results, columns=["LN_ID","label_ene","predict_ene"])
else:
    label_predictions = label_predictions.reshape(label_predictions.shape[0],label_predictions.shape[2])
    results = np.column_stack((ids,labels,label_predictions[:,0],label_predictions[:,1]))
    results_df = pd.DataFrame(data=results, columns=["LN_ID","label_pos","label_ene","predict_pos","predict_ene"])


labels_df = pd.read_csv('/media/bhkann/HN_RES1/HN_DL/YST_LN_LABEL.csv')
results_df = pd.merge(results_df, labels_df, how="left", on="LN_ID")
print(results_df)

#list(set(results_df.LN_ID) - set(labels_df.LN_ID))

print("rows with nan results: ", results_df[results_df['label_ene']=='nan'])
#list(results_df.label_ene)
#df.to_csv(os.path.join(SAVE_DIR, "testresults_{}_{}.csv".format(DATASET,NAME))) 
#results_df = results_df.loc[(results_df['label_ene'] == 1) & (results_df['label_ene'] == 0)]
#results_df = results_df.dropna(subset=['label_ene'])
#results_df = results_df[results_df['predict_ene'].notna()]
if DATASET == 'E3311':
    results_df['macro'] = np.where(results_df['MacroENE']==1,1,0)

results_certain = results_df[results_df['CertaintyECE']==1]


roc_test_ece = roc_auc_score(results_df.label_ene, results_df.predict_ene)
print('\rroc-auc_TEST:', str(round(roc_test_ece,4))+'\n')

roc_test_ece = roc_auc_score(results_certain.label_ene, results_certain.predict_ene)
print('\rroc-auc_TEST certain:', str(round(roc_test_ece,4))+'\n')

if DATASET == 'E3311':
    results_macro = results_certain[results_certain['MacroENE']!=0] #results_certain.loc(results_certain['MacroENE']==1, 1,0)
    roc_test_ece = roc_auc_score(results_macro.label_ene, results_macro.predict_ene)
    print('\rroc-auc_TEST certain macro:', str(round(roc_test_ece,4))+'\n')
    roc_test_ece = roc_auc_score(results_certain.macro, results_certain.predict_ene)
    print('\rroc-auc_TEST certain macro=1 else =0:', str(round(roc_test_ece,4))+'\n')
#roc_test_ece
#try:
#roc_test_ece = roc_auc_score(self.y_test[:,1], y_pred_test[:,1])

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

#best_threshold = 0.55
predictecebin = np.copy(y_prob)

predictecebin[predictecebin >= best_threshold] = 1
predictecebin[predictecebin < best_threshold] = 0
print(classification_report(results_certain['label_ene'].astype('float'), predictecebin))
print('accuracy: ', accuracy_score(results_certain['label_ene'].astype('float'), predictecebin))

#sklearn.metrics.recall_score

print("Using Threshold 0.3")
predictecebin = np.copy(y_prob)
predictecebin[predictecebin >= 0.3] = 1
predictecebin[predictecebin < 0.3] = 0
print(classification_report(results_certain['label_ene'].astype('float'), predictecebin))
print('accuracy: ', accuracy_score(results_certain['label_ene'].astype('float'), predictecebin)) 

if DATASET == 'E3311':
    print("excluding micro ENE")
    #best_threshold = 0.70
    predictecebin = np.array(results_macro.predict_ene).astype('float') 
    predictecebin[predictecebin >= best_threshold] = 1
    predictecebin[predictecebin < best_threshold] = 0
    print(classification_report(results_macro['label_ene'].astype('float'), predictecebin))
    print('macro accuracy: ', accuracy_score(results_macro['label_ene'].astype('float'), predictecebin)) 


# populate df
if SAVE_CSV:
    df = pd.DataFrame.from_dict(results_df)
    results_df.to_csv(os.path.join(SAVE_DIR, "testresults_{}_{}.csv".format(DATASET,NAME))) # Can choose lab drive vs HPC to save
    results_df.to_csv(os.path.join('/media/bhkann/HN_RES1/HN_DL/E3311_DL/E3311_analysis', "testresults_{}_{}.csv".format(DATASET,NAME)))
    '''