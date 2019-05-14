# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:31:51 2018

@author: prant
"""

import os
import numpy as np
import cv2
from tqdm import tqdm

# directory
img_dir = "Uniform_XR_WRIST_HIST"
img_classes=["Abnormal","Normal"]

img_size = 256

if img_size == 256:
    training_data=[]
    def create_training_data():
     
        for clas in tqdm(img_classes):
                path=os.path.join(img_dir,clas)
                #index no start from 0 1 2 3
                print(clas)
                
                class_num=img_classes.index(clas)
                print(class_num)
                
                for img in os.listdir(path):
                    try:
                        img_array=cv2.imread(os.path.join(path,img))
                        new_array=cv2.resize(img_array,(img_size,img_size))
                        training_data.append([new_array,class_num])
                    except Exception as e:
                        pass
        
           
    create_training_data()
    print(len(training_data))
    
    xs=[]
    ys=[]
    
    for features,labels in tqdm(training_data):
        xs.append(features)
        ys.append(labels)
    
    xs=np.array(xs).reshape(-1,img_size,img_size,3)
    ys=np.array(ys).reshape(-1,1)
    
    import pickle
    pickle_out=open("Uniform_X_WRIST_HIST_256.pickle","wb")
    pickle.dump(xs,pickle_out)
    pickle_out.close()
    
    pickle_out=open("Uniform_Y_WRIST_HIST_256.pickle","wb")
    pickle.dump(ys,pickle_out)
    pickle_out.close()
    
elif img_size == 420:
    training_data=[]
    def create_training_data():
     
        for clas in tqdm(img_classes):
                path=os.path.join(img_dir,clas)
                #index no start from 0 1 2 3
                print(clas)
                
                class_num=img_classes.index(clas)
                print(class_num)
                
                for img in os.listdir(path):
                    try:
                        img_array=cv2.imread(os.path.join(path,img))
                        new_array=cv2.resize(img_array,(img_size,img_size))
                        training_data.append([new_array,class_num])
                    except Exception as e:
                        pass
        
           
    create_training_data()
    print(len(training_data))
    
    xs=[]
    ys=[]
    
    for features,labels in tqdm(training_data):
        xs.append(features)
        ys.append(labels)
    
    xs=np.array(xs).reshape(-1,img_size,img_size,3)
    ys=np.array(ys).reshape(-1,1)
    
    import pickle
    pickle_out=open("Uniform_X_WRIST_HIST_420.pickle","wb")
    pickle.dump(xs,pickle_out)
    pickle_out.close()
    
    pickle_out=open("Uniform_Y_WRIST_HIST_420.pickle","wb")
    pickle.dump(ys,pickle_out)
    pickle_out.close()

