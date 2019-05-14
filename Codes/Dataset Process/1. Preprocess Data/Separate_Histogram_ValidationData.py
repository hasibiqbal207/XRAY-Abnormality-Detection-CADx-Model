# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 14:53:47 2018

@author: prant
"""

import csv
import os
import time
import cv2
import numpy as np

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
        file_path = os.path.join(directory[0], filename)
        
        # Histogram Process Start
        img = cv2.imread(file_path)
        
        ######### normalization
        normalizedImg = np.zeros((800, 800))
        normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)

        ########## histogram equalization        
        hist_img = normalizedImg

        try:
            kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            cv2.filter2D(hist_img, -1, kernel)
            hist,bins = np.histogram(hist_img.flatten(),256,[0,256])
            
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max()/ cdf.max()
        except:
            None
        
        
        if ( "XR_ELBOW" in directory[0] ):
            if directory[1] == '0':
                dir_his = "Dataset_Histogram/XR_ELBOW/Validation/Abnormal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)                    
                    
                his_file_name = "abnormal_XR_ELBOW_hist_" + str(XR_ELBOW) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)                

            else:
                dir_his = "Dataset_Histogram/XR_ELBOW/Validation/Normal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)                    
                    
                his_file_name = "normal_XR_ELBOW_hist_" + str(XR_ELBOW) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)
            
            XR_ELBOW += 1
                
        elif ( "XR_FINGER" in directory[0] ):
            if directory[1] == '0':
                dir_his = "Dataset_Histogram/XR_FINGER/Validation/Abnormal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)
                
                his_file_name = "abnormal_XR_FINGER_hist_" + str(XR_FINGER) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)
            else:
                dir_his = "Dataset_Histogram/XR_FINGER/Validation/Normal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)
                
                his_file_name = "normal_XR_FINGER_hist_" + str(XR_FINGER) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)
                    
            XR_FINGER += 1
            
        elif ( "XR_FOREARM" in directory[0] ):
            if directory[1] == '0':
                dir_his = "Dataset_Histogram/XR_FOREARM/Validation/Abnormal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)                    
                                
                his_file_name = "abnormal_XR_FOREARM_hist_" + str(XR_FOREARM) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img) 
            else:
                dir_his = "Dataset_Histogram/XR_FOREARM/Validation/Normal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)               

                his_file_name = "normal_XR_FOREARM_hist_" + str(XR_FOREARM) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)
            
            XR_FOREARM += 1
            
        elif ( "XR_HAND" in directory[0] ):
            if directory[1] == '0':
                dir_his = "Dataset_Histogram/XR_HAND/Validation/Abnormal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)                   
                
                his_file_name = "abnormal_XR_HAND_hist_" + str(XR_HAND) + ".png"
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)
            else:
                dir_his = "Dataset_Histogram/XR_HAND/Validation/Normal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)                  
                
                his_file_name = "normal_XR_HAND_hist_" + str(XR_HAND) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)
            
            XR_HAND += 1
            
        elif ( "XR_HUMERUS" in directory[0] ):
            if directory[1] == '0':
                dir_his = "Dataset_Histogram/XR_HUMERUS/Validation/Abnormal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)                   
                
                his_file_name = "abnormal_XR_HUMERUS_hist_" + str(XR_HUMERUS) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)
            else:
                dir_his = "Dataset_Histogram/XR_HUMERUS/Validation/Normal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)                    
                        
                his_file_name = "normal_XR_HUMERUS_hist_" + str(XR_HUMERUS) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)
            
            XR_HUMERUS += 1
            
        elif ( "XR_SHOULDER" in directory[0] ):
            if directory[1] == '0':
                dir_his = "Dataset_Histogram/XR_SHOULDER/Validation/Abnormal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)                    
                
                his_file_name = "abnormal_XR_SHOULDER_hist_" + str(XR_SHOULDER) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)   
            else:
                dir_his = "Dataset_Histogram/XR_SHOULDER/Validation/Normal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)                   
                
                his_file_name = "normal_XR_SHOULDER_hist_" + str(XR_SHOULDER) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)   
            
            XR_SHOULDER += 1
            
        elif ( "XR_WRIST" in directory[0] ):
            if directory[1] == '0':
                dir_his = "Dataset_Histogram/XR_WRIST/Validation/Abnormal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)                    
                
                his_file_name = "abnormal_XR_WRIST_hist_" + str(XR_WRIST) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)
            else:
                dir_his = "Dataset_Histogram/XR_WRIST/Validation/Normal"
                if not os.path.exists(dir_his):
                    os.makedirs(dir_his)                   
                
                his_file_name = "normal_XR_WRIST_hist_" + str(XR_WRIST) + ".png"                
                cv2.imwrite(os.path.join(dir_his , his_file_name), hist_img)
            
            XR_WRIST += 1
            
        time.sleep(.1)
file.close()