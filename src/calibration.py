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
DATASET='YST' #
DIR = '/OUTPUT/' # Output Folder for INTERNAL VALIDATION COHORT
FOLDER_NAME = "1_ene_multi_dualnet_nrrd_0.001b16w1.0sigmoid_OVTrue_-175_275_CERTFalse_20220208-1531"
df_certain = pd.read_csv(DIR + FOLDER_NAME + '/testresults_' + DATASET + '_' + FOLDER_NAME + '.csv') # + '_kfold_ensemble.csv')

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

## LOAD TEST DATA ##
results_e3311 = pd.read_csv(DIR + FOLDER_NAME + '/testresults_E3311_' + FOLDER_NAME + '.csv') # )
dfe3311_certain = results_e3311[results_e3311["CertaintyECE"] == 1]
y_ext = np.array(dfe3311_certain['label_ene']).astype('float')
pred_ext = np.array(dfe3311_certain.predict_ene).astype('float') #change based on class

temperature = LogisticCalibration()  # PLATT SCALING #
temperature.fit(pred_val, y_val)
calibrated = temperature.transform(pred_ext)

dfe3311_certain['calibrated_ene']=calibrated
dfe3311_certain[['predict_ene','calibrated_ene']]

## SAVE CALIBRATED PROBABILITIES
dfe3311_certain.to_csv(DIR + FOLDER_NAME + '/testresults_E3311_' + FOLDER_NAME + '_calibrated.csv')


n_bins = 10
diagram = ReliabilityDiagram(n_bins)
diagram.plot(pred_ext, y_ext).show()  # visualize miscalibration of uncalibrated
diagram.plot(calibrated, y_ext).show()   # visualize calibration of calibrated


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

plt.savefig('XXXX.png', dpi=600, facecolor='w', edgecolor='w',
        orientation='portrait', format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
plt.show()


from sklearn.metrics import f1_score, confusion_matrix, precision_score, accuracy_score, hamming_loss, classification_report, roc_auc_score, matthews_corrcoef

roc_test_ece = roc_auc_score(y_ext, calibrated)
print('\rroc-auc_TEST certain macro=1 else =0:', str(round(roc_test_ece,4))+'\n')

roc_auc_score(y_ext, pred_ext)