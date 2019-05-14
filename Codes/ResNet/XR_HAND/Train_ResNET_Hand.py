# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:06:59 2018

@author: prant
"""

import time
import pickle

from keras import Model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.utils import to_categorical
from keras import optimizers

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import applications

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

t1=time.time()
x_train = pickle.load(open("Splited_X_Train_Hand_HIST_256.pickle","rb"))
y_train = pickle.load(open("Splited_Y_Train_Hand_HIST_256.pickle","rb"))
y_train = to_categorical(y_train,2)
t2=time.time()
print("time to load :",(t2-t1))

t1=time.time()
x_test = pickle.load(open("Splited_X_Test_Hand_HIST_256.pickle","rb"))
y_test = pickle.load(open("Splited_Y_Test_Hand_HIST_256.pickle","rb"))
y_test_ROC = y_test
y_test = to_categorical(y_test,2)
t2=time.time()
print("time to load :",(t2-t1))


epoch = 50

#set early stopping criteria
pat = 10 #this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)
#define the model checkpoint callback -> this will keep on saving the model as a physical file
model_checkpoint = ModelCheckpoint('Hand_ResNET_Model_by_Machine.h5', verbose=1, save_best_only=True)

def final_model():
    img_width, img_height = 256, 256
    model = applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
 
     # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers[:5]:
        layer.trainable = False
    
    #Adding custom Layers 
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation="softmax")(x)
    
    # creating the final model 
    model_final = Model(inputs = model.input, output = predictions)
    
    # compile the model 
    model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
    return model_final


def fit_and_evaluate(X_Train, X_Test, Y_Train, Y_Test, y_test_ROC):
    
    model = None
    model = final_model()
    
    results = model.fit(X_Train, Y_Train, batch_size=8, epochs = epoch, callbacks=[early_stopping, model_checkpoint], verbose=1, validation_split=0.05)
    model.save('Hand_ResNET50_Model.h5')
    
    ################## Result Evaluation Start ###################
    
    score = model.evaluate(X_Test, Y_Test, batch_size=8)
    print(score)
    
    # Make prediction from model
    predictions = model.predict(X_Test)
    Y_Pred = (predictions > 0.5)
    Y_Test = (Y_Test > 0.5)
    
    
    ###########  ROC Curve Generate    
    
    probs = predictions
    
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    
    # calculate AUC
    auc = roc_auc_score(y_test_ROC, probs)
    auc = '%.3f' % auc
    print('AUC: ' +  auc)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test_ROC, probs)
    
    pyplot.figure()
    # plot no skill
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC Curve ( Area = ' + auc +' )')
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    
    pyplot.plot(fpr, tpr, marker='.')
    # show the plot
    pyplot.title('Receiver Operating Characteristic ( ROC )')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()
    pyplot.savefig('ROC_' + auc + '.png')
        
                        ############################################
                        #       Overal: Abnormal + Normal          #
                        ############################################
    Y_Pred_all = Y_Pred
    print(Y_Pred_all.shape)
    Y_Pred_all = Y_Pred_all.reshape(-1,1)
    
    Y_Test_all = Y_Test
    print(Y_Test_all.shape)
    Y_Test_all = Y_Test_all.reshape(-1,1)
    
    file = open("Result_of_Model.txt","w") 
    
    
    con_mat = confusion_matrix(Y_Test_all, Y_Pred_all)
    print("Confusion Matrix => Overal: Abnormal + Normal : \n" + str(con_mat))
    file.write("Confusion Matrix => Overal: Abnormal + Normal : \n" + str(con_mat) + "\n\n")
    
                        ############################################
                        #               Abnormal                   #
                        ############################################
    Y_Pred_abn = Y_Pred
    print(Y_Pred_abn.shape)
    Y_Pred_abn = Y_Pred[:, 0]
    
    Y_Test_abn = Y_Test
    print(Y_Test_abn.shape)
    Y_Test_abn = Y_Test_abn[:, 0]
    
    con_mat = confusion_matrix(Y_Test_abn, Y_Pred_abn)
    print("Confusion Matrix => Abnormal : \n" + str(con_mat))
    file.write("Confusion Matrix => Abnormal : \n" + str(con_mat) + "\n\n")
    
                        ############################################
                        #               Normal                    #
                        ############################################
    Y_Pred_nor = Y_Pred
    print(Y_Pred_nor.shape)
    Y_Pred_nor = Y_Pred[:, 1]
    
    Y_Test_nor = Y_Test
    print(Y_Test_nor.shape)
    Y_Test_nor = Y_Test_nor[:, 1]
    
    con_mat = confusion_matrix(Y_Test_nor, Y_Pred_nor)
    print("Confusion Matrix => Normal : \n" + str(con_mat))
    file.write("Confusion Matrix => Normal : \n" + str(con_mat))
    
    file.close() 
    
    ################## Result Evaluation END ###################
    print('\n\n\n')
    
    return results


model_history = [] 
model_history.append(fit_and_evaluate(x_train, x_test, y_train, y_test, y_test_ROC))

plt.figure()
plt.title('Accuracies vs Epochs On Training Data')
plt.plot(model_history[0].history['acc'], label='Training Data')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('Accuracies_vs_Epochs_training.png')

plt.figure()
plt.title('Accuracies vs Epochs On Validation Data')
plt.plot(model_history[0].history['val_acc'], label='Validation Data')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('Accuracies_vs_Epochs_validation.png')

plt.figure()
plt.title('Train Accuracy vs Validation Accuracy')
plt.plot(model_history[0].history['acc'], label='Training Accuracy', color='black')
plt.plot(model_history[0].history['val_acc'], label='Validation Accuracy', color='black', linestyle = "dashdot")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('Train_vs_Val.png')