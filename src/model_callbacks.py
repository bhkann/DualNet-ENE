#ENE MODEL #
import keras
import numpy as np
import pandas as pd 
from statistics import median
from keras import backend as K
from sklearn.metrics import f1_score, confusion_matrix, precision_score, accuracy_score, hamming_loss, classification_report, roc_auc_score, matthews_corrcoef

#from metrics import dice

class model_callbacks(keras.callbacks.Callback):

    def __init__(self, model, RUN, dir_name, val_data):
        self.model_to_save = model
        self.run = RUN
        self.dir_name = dir_name
        self.val_data = val_data
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = 1000
        self.val_dice_scores = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        self.val_losses.append(val_loss)
        lr = float(K.get_value(self.model.optimizer.lr))
        # lr = self.model.optimizer.lr
        self.learning_rates.append(lr)
        # save model
        if val_loss < self.best_val_loss:
            self.model_to_save.save(self.dir_name + '/{}.h5'.format(self.run))
            self.best_val_loss = val_loss
            print("model saved.")
        '''    
        for i in range(0,len(self.val_data[0])):
            image = self.val_data[0][i]
            image_small = self.val_data[1][i]
            label = self.val_data[2][i]
            print('image shape:', image.shape, image_small.shape)
            #label = label.reshape(1,*label.shape)
            print('label truth:',label)
            label_predict = self.model_to_save.predict([image.reshape(*image.shape,1),image_small.reshape(*image_small.shape,1)],batch_size=1) #.reshape(*image.shape,1),image_small.reshape(*image_small.shape,1)]) #.reshape(1,*image.shape))
            print("prediction, truth", np.array([label_predict,label]))
            #label_predict = np.squeeze(label_predict)
            #label_predict[label_predict<0.5]=0
            #label_predict[label_predict>=0.5]=1
            
            #dice_score = dice(np.squeeze(label), label_predict)
            #self.val_dice_scores.append(dice_score)
        #med_dice = median(self.val_dice_scores)
        #print("validation average dice score: ", med_dice)
        '''    
        # save logs (overwrite)
        np.save(self.dir_name + '/{}_loss.npy'.format(self.run), self.losses)
        np.save(self.dir_name + '/{}_val_loss.npy'.format(self.run), self.val_losses)
        np.save(self.dir_name + '/{}_lr.npy'.format(self.run), self.learning_rates)
        
    def on_train_end(self, logs):
        self.model_to_save.save(self.dir_name + '/{}_final.h5'.format(self.run))


############
### Define AUC Callback
class roc_callback(keras.callbacks.Callback):
    def __init__(self,validation_data, test_data, dir_name, modelname, classifier, activator):
        
        self.x_test = test_data[0]
        self.y_test = test_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.dir_name = dir_name
        self.modelname = modelname
        self.classifier = classifier
        self.activator = activator
    def on_train_begin(self, logs={}):
        self.aucspos = []
        self.aucsece = []
        self.epochcurrent = []
        self.youden_max = []
        self.threshold_best = []
        return
    
    def on_train_end(self, logs={}):
        return
    
    def on_epoch_begin(self, epoch, logs={}):
        return
    
    def on_epoch_end(self, epoch, logs={}):        
        #y_pred = self.model.predict(self.x)
        #roc = roc_auc_score(self.y, y_pred)      
        y_pred_val = self.model.predict(self.x_val)
        y_pred_test = self.model.predict(self.x_test)
        if (self.classifier == 'ene' or self.classifier == 'pos') and self.activator == 'sigmoid':
            y_val = self.y_val[:,0]
            y_pred_val = y_pred_val[:,0]
            y_test = self.y_test[:,0]
            y_pred_test = y_pred_test[:,0]
            roc_val_pos = 0
            roc_test_pos = 0
            roc_val_ece = roc_auc_score(y_val, y_pred_val)
            y_prob = np.array(y_pred_test)
        elif self.activator == 'softmax':
            y_val = self.y_val[:,1]
            y_pred_val = y_pred_val[:,1]
            y_test = self.y_test[:,1]
            y_pred_test = y_pred_test[:,1]            
            roc_val_pos = 0
            roc_test_pos = 0
            roc_val_ece = roc_auc_score(y_val, y_pred_val)
            y_prob = np.array(y_pred_test)
            #roc_test_ece = roc_auc_score(self.y_val,self.model.predict(self.x_val))
        elif self.classifier == 'multi' or self.classifier == 'small':
            #y_val = self.y_val[:,1]
            #y_pred_val = y_pred_val[:,1]
            #y_pred_test = y_pred_test[:,1]
            roc_val_pos = roc_auc_score(self.y_val[:,0], y_pred_val[:,0])      
            roc_test_pos = roc_auc_score(self.y_test[:,0], y_pred_test[:,0])
            roc_val_ece = roc_auc_score(self.y_val[:,1], y_pred_val[:,1])      
            roc_test_ece = roc_auc_score(self.y_test[:,1], y_pred_test[:,1])
            y_prob = y_pred_test[:,1] #change based on class
            y_test = self.y_test[:,1]
            #y_pred_test = y_pred_test[:,1]
        self.aucsece.append(roc_val_ece)
        self.aucspos.append(roc_val_pos)
        
        #print('label truth, label prediction:', np.around(np.column_stack((self.y_val,y_pred_val)),decimals=2))
        
        print("current max AUC and epoch #: ", max(self.aucsece, default=0), self.aucsece.index(max(self.aucsece)))
        if roc_val_ece >= max(self.aucsece, default=0):
            self.model.save('/media/bhkann/HN_RES1/' + self.dir_name + '/' + 'bestauc.h5')
            print("best val auc so far. MODEL SAVED.")
        self.epochcurrent.append(epoch)
        print("auc appended to list.")

        print('\rroc-auc_VAL pos, ece:', str(round(roc_val_pos,4)) + '  ' + str(round(roc_val_ece,4))+'\n')
        print('\rroc-auc_TEST ece:', str(round(roc_test_ece,4))+'\n')
        #roc_test_ece
        #try:
        #roc_test_ece = roc_auc_score(self.y_test[:,1], y_pred_test[:,1])
        #print('\rroc-auc_TEST 1:', str(round(roc_test_ece,4))+'\n')
        threshold = np.arange(0.1,0.9,0.1)
        acc = []
        accuracies = []
        YI = []
        Youdens = []
        best_threshold = np.zeros(y_prob.shape)
        for j in threshold:
            y_pred = [1.0 if prob>=j else 0 for prob in y_prob]
            if self.activator == 'sigmoid':
                cm = confusion_matrix(y_test, y_pred) # ##True, predicted
            elif self.activator == 'softmax':
                cm = confusion_matrix(y_test, y_pred)
            TN, FP, FN, TP = cm.ravel().astype('float32')
            if TP + FN == 0 or TN + FP == 0:
                TPR, TNR = 0, 0
            else:
                TPR = TP/(TP+FN)
                # Specificity or true negative rate
                TNR = TN/(TN+FP) 
            Youden = TPR + TNR - 1
            Youdens.append(Youden)  #change metric based on class using
        Youdens = np.array(Youdens)
        index = np.where(Youdens==Youdens.max()) 
        #YI.append(Youdens.max()) 
        try:
            best_threshold = threshold[index[0][0]]
            print("best threshold YI: ", best_threshold)
        except: 
            print('best threshold unable to be calculated, using 0.5')
            best_threshold = 0.5
            #ECE 0.1, 0.8 ;  for multi-label: 0.1, 0.6, 0.8
        predictecebin = np.copy(y_prob)
        predictecebin[predictecebin >= best_threshold] = 1
        predictecebin[predictecebin < best_threshold] = 0
        print(classification_report(y_test, predictecebin))
        print('accuracy: ', accuracy_score(y_test, predictecebin))     

        self.youden_max.append(Youdens.max())
        self.threshold_best.append(best_threshold)

        df = pd.DataFrame(list(zip(self.epochcurrent,self.aucspos,self.aucsece,self.youden_max,self.threshold_best)),columns=['epoch','auc-pos','auc-ene','best-youden','best-threshold'])
        df.to_csv('/media/bhkann/HN_RES1/' + self.dir_name + '/' + self.modelname + '_epoch_aucs_youdens.csv')
        return
        
    def on_batch_begin(self, batch, logs={}):
        return
    
    def on_batch_end(self, batch, logs={}):
        return   

