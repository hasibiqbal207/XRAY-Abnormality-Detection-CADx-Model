# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 12:04:30 2018

@author: prant
"""

import csv
import os
import time

file = open('MURA-v1.1/valid_labeled_studies.csv')
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
                dir = "Dataset_Single/XR_ELBOW/Validation/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_ELBOW/Validation/Abnormal/abnormal_XR_ELBOW" + str(XR_ELBOW) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_ELBOW/Validation/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_ELBOW/Validation/Normal/normal_XR_ELBOW" + str(XR_ELBOW) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
                
        elif ( "XR_FINGER" in directory[0] ):
            XR_FINGER += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_FINGER/Validation/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_FINGER/Validation/Abnormal/abnormal_XR_FINGER" + str(XR_FINGER) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_FINGER/Validation/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_FINGER/Validation/Normal/normal_XR_FINGER" + str(XR_FINGER) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            
        elif ( "XR_FOREARM" in directory[0] ):
            XR_FOREARM += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_FOREARM/Validation/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_FOREARM/Validation/Abnormal/abnormal_XR_FOREARM" + str(XR_FOREARM) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_FOREARM/Validation/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_FOREARM/Validation/Normal/normal_XR_FOREARM" + str(XR_FOREARM) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            
        elif ( "XR_HAND" in directory[0] ):
            XR_HAND += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_HAND/Validation/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_HAND/Validation/Abnormal/abnormal_XR_HAND" + str(XR_HAND) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_HAND/Validation/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_HAND/Validation/Normal/normal_XR_HAND" + str(XR_HAND) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            
        elif ( "XR_HUMERUS" in directory[0] ):
            XR_HUMERUS += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_HUMERUS/Validation/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_HUMERUS/Validation/Abnormal/abnormal_XR_HUMERUS" + str(XR_HUMERUS) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_HUMERUS/Validation/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_HUMERUS/Validation/Normal/normal_XR_HUMERUS" + str(XR_HUMERUS) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            
        elif ( "XR_SHOULDER" in directory[0] ):
            XR_SHOULDER += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_SHOULDER/Validation/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_SHOULDER/Validation/Abnormal/abnormal_XR_SHOULDER" + str(XR_SHOULDER) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_SHOULDER/Validation/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_SHOULDER/Validation/Normal/normal_XR_SHOULDER" + str(XR_SHOULDER) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            
        elif ( "XR_WRIST" in directory[0] ):
            XR_WRIST += 1
            if directory[1] == '0':
                dir = "Dataset_Single/XR_WRIST/Validation/Abnormal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_WRIST/Validation/Abnormal/abnormal_XR_WRIST" + str(XR_WRIST) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            else:
                dir = "Dataset_Single/XR_WRIST/Validation/Normal"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                temp = "Dataset_Single/XR_WRIST/Validation/Normal/normal_XR_WRIST" + str(XR_WRIST) + ".png"
                os.rename(os.path.join(directory[0], filename), temp)
            
        time.sleep(.1)


file.close()