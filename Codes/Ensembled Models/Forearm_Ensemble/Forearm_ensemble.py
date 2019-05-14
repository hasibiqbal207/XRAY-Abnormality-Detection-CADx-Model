from keras.models import load_model
from keras import optimizers
import pickle
from keras.utils import to_categorical
import numpy as np

model_vgg = load_model('Forearm_VGG_Model_by_Machine.h5')
model_vgg.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

model_res = load_model('Forearm_ResNET_Model_by_Machine.h5')
model_res.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


X_Test = pickle.load(open("Splited_X_Test_Forearm_HIST_256.pickle","rb"))
Y_Test = pickle.load(open("Splited_Y_Test_Forearm_HIST_256.pickle","rb"))
Y_Test = to_categorical(Y_Test,2)

# Confusion Matrics Values
TP = 0
FP = 0
FN = 0
TN = 0

# Used for situation 50/50
drop = 0

# Variable
Type = "Forearm"
vgg_weight = 0.8636
res_weight = 0.1364

lenth = len(Y_Test)
for i in range(lenth):
    img = np.reshape(X_Test[i],[1,256,256,3])
    
    # Temporary Variable
    class_label = 0
    true_label = int(Y_Test[i].item(1))  
    
    classifier1 = model_vgg.predict(img)
    classifier2 = model_res.predict(img)
    
    classifier_per_0 = (classifier1.item(0)*vgg_weight + classifier2.item(0)*res_weight)
    classifier_per_1 = (classifier1.item(1)*vgg_weight + classifier2.item(1)*res_weight)

#     Predicted Class Label Selection
    if(classifier_per_0 > classifier_per_1):
        class_label = 0
    elif(classifier_per_0 < classifier_per_1):
        class_label = 1
    else:
        drop += 1
        continue

    # Confusion Matrics Value Estimation
    if(class_label == true_label == 0):
        TP += 1
    elif(class_label == true_label == 1):
        TN += 1
    elif(true_label == 0 and class_label == 1):
        FN += 1
    elif(true_label == 1 and class_label == 0):
        FP += 1
    else:
        None

Accuracy = ( TP + TN ) / (TP+FP+TN+FN)

True_rate = ((TP+FP)/(TP+FP+FN+TN))*((TP+FN)/(TP+FP+FN+TN))
False_rate = ((TN+FP)/(TP+FP+FN+TN))*((TN+FN)/(TP+FP+FN+TN))
expected = True_rate + False_rate
Cohen_kappa = (Accuracy - expected)/(1 - expected -.03)

file = open(Type  + "_Result_of_Ensemble.txt","w")

file.write("TP => " + str(TP) + "\tFP => " + str(FP) +"\nFN => " + str(FN) +"\tTN => " + str(TN) + '\n\n')

file.write("Accuracy => " + str(Accuracy) + '\n\n')

file.write("Cohen's kappa => " + str(Cohen_kappa)) 

file.close()