# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 14:53:47 2018

@author: prant
"""

import csv
import os
import time

file = open('MURA-v1.1/train_labeled_studies.csv')
file_readable = csv.reader(file)

XR_ELBOW = 0
XR_FINGER = 0
XR_FOREARM = 0
XR_HAND = 0
XR_HUMERUS = 0
XR_SHOULDER = 0
XR_WRIST = 0

for directory in file_readable:  
    for filename in os.listdir(directory[0]):
        os.path.join(directory[0], filename)
        if ( "XR_ELBOW" in directory[0] ):
            XR_ELBOW += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_ELBOW/Train/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_ELBOW/Train/Abnormal/abnormal_XR_ELBOW" + str(XR_ELBOW) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_ELBOW/Train/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_ELBOW/Train/Normal/normal_XR_ELBOW" + str(XR_ELBOW) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
                
        elif ( "XR_FINGER" in directory[0] ):
            XR_FINGER += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_FINGER/Train/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_FINGER/Train/Abnormal/abnormal_XR_FINGER" + str(XR_FINGER) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_FINGER/Train/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_FINGER/Train/Normal/normal_XR_FINGER" + str(XR_FINGER) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            
        elif ( "XR_FOREARM" in directory[0] ):
            XR_FOREARM += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_FOREARM/Train/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_FOREARM/Train/Abnormal/abnormal_XR_FOREARM" + str(XR_FOREARM) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_FOREARM/Train/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_FOREARM/Train/Normal/normal_XR_FOREARM" + str(XR_FOREARM) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            
        elif ( "XR_HAND" in directory[0] ):
            XR_HAND += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_HAND/Train/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_HAND/Train/Abnormal/abnormal_XR_HAND" + str(XR_HAND) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_HAND/Train/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_HAND/Train/Normal/normal_XR_HAND" + str(XR_HAND) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            
        elif ( "XR_HUMERUS" in directory[0] ):
            XR_HUMERUS += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_HUMERUS/Train/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_HUMERUS/Train/Abnormal/abnormal_XR_HUMERUS" + str(XR_HUMERUS) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_HUMERUS/Train/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_HUMERUS/Train/Normal/normal_XR_HUMERUS" + str(XR_HUMERUS) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            
        elif ( "XR_SHOULDER" in directory[0] ):
            XR_SHOULDER += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_SHOULDER/Train/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_SHOULDER/Train/Abnormal/abnormal_XR_SHOULDER" + str(XR_SHOULDER) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_SHOULDER/Train/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_SHOULDER/Train/Normal/normal_XR_SHOULDER" + str(XR_SHOULDER) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            
        elif ( "XR_WRIST" in directory[0] ):
            XR_WRIST += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_WRIST/Train/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_WRIST/Train/Abnormal/abnormal_XR_WRIST" + str(XR_WRIST) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_WRIST/Train/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_WRIST/Train/Normal/normal_XR_WRIST" + str(XR_WRIST) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            
        time.sleep(.1)


file.close()