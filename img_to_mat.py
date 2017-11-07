#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: bhumihar
"""
import numpy as np
import cv2 
import os 
from matplotlib import pyplot
from scipy.misc import toimage



            ############ Code to convert train image into matrix ###############
a_img_list = os.listdir('/home/bhumihar/Programming/Python/opencv/sample/Marcel-Train/A')
b_img_list = os.listdir('/home/bhumihar/Programming/Python/opencv/sample/Marcel-Train/B')
c_img_list = os.listdir('/home/bhumihar/Programming/Python/opencv/sample/Marcel-Train/C')
f_img_list = os.listdir('/home/bhumihar/Programming/Python/opencv/sample/Marcel-Train/Five')
p_img_list = os.listdir('/home/bhumihar/Programming/Python/opencv/sample/Marcel-Train/Point')
v_img_list = os.listdir('/home/bhumihar/Programming/Python/opencv/sample/Marcel-Train/V')

A_image = np.zeros((np.shape(a_img_list)[0],50,50,3),dtype=np.float)
A_value = np.zeros((np.shape(a_img_list)[0]),dtype=np.float)
i =0 ;
for img in a_img_list :
    file_name = "/home/bhumihar/Programming/Python/opencv/sample/Marcel-Train/A/"+img ;
    image = cv2.imread(file_name)
    if(image.shape!=(50,50,3)) :
        image = cv2.resize(image, (50, 50))
    A_image[i,:,:,:] = image
    A_value[i] = 1;
    i = i+1
    
B_image = np.zeros((np.shape(b_img_list)[0],50,50,3),dtype=np.float)
B_value = np.zeros((np.shape(b_img_list)[0]),dtype=np.float)
i =0 ;
for img in b_img_list :
    file_name = "/home/bhumihar/Programming/Python/opencv/sample/Marcel-Train/B/"+img ;
    image = cv2.imread(file_name)
    if(image.shape!=(50,50,3)) :
        image = cv2.resize(image, (50, 50))
    B_image[i,:,:,:] = image
    B_value[i] = 1;
    i = i+1
    
C_image = np.zeros((np.shape(c_img_list)[0],50,50,3),dtype=np.float)
C_value = np.zeros((np.shape(c_img_list)[0]),dtype=np.float)
i =0 ;
for img in c_img_list :
    file_name = "/home/bhumihar/Programming/Python/opencv/sample/Marcel-Train/C/"+img ;
    image = cv2.imread(file_name)
    if(image.shape!=(50,50,3)) :
        image = cv2.resize(image, (50, 50))
    C_image[i,:,:,:] = image
    C_value[i] = 1;
    i = i+1
    
F_image = np.zeros((np.shape(f_img_list)[0],50,50,3),dtype=np.float)
F_value = np.zeros((np.shape(f_img_list)[0]),dtype=np.float)
i =0 ;
for img in f_img_list :
    file_name = "/home/bhumihar/Programming/Python/opencv/sample/Marcel-Train/Five/"+img ;
    image = cv2.imread(file_name)
    if(image.shape!=(50,50,3)) :
        image = cv2.resize(image, (50, 50))
    F_image[i,:,:,:] = image
    F_value[i] = 1;
    i = i+1

P_image = np.zeros((np.shape(p_img_list)[0],50,50,3),dtype=np.float)
P_value = np.zeros((np.shape(p_img_list)[0]),dtype=np.float)
i =0 ;
for img in p_img_list :
    file_name = "/home/bhumihar/Programming/Python/opencv/sample/Marcel-Train/Point/"+img ;
    image = cv2.imread(file_name)
    if(image.shape!=(50,50,3)) :
        image = cv2.resize(image, (50, 50))
    P_image[i,:,:,:] = image
    P_value[i] = 1;
    i = i+1
    
V_image = np.zeros((np.shape(v_img_list)[0],50,50,3),dtype=np.float)
V_value = np.zeros((np.shape(v_img_list)[0]),dtype=np.float)
i =0 ;
for img in v_img_list :
    file_name = "/home/bhumihar/Programming/Python/opencv/sample/Marcel-Train/V/"+img ;
    image = cv2.imread(file_name)
    if(image.shape!=(50,50,3)) :
        image = cv2.resize(image, (50, 50))
    V_image[i,:,:,:] = image
    V_value[i] = 1;
    i = i+1
    
print(np.shape(A_image))
print(np.shape(B_image))
print(np.shape(C_image))
print(np.shape(F_image))
print(np.shape(P_image))
print(np.shape(V_image))

X_train = np.zeros((np.shape(A_image)[0] +np.shape(B_image)[0] +np.shape(C_image)[0] 
                   +np.shape(F_image)[0] +np.shape(P_image)[0] +np.shape(V_image)[0],50,50,3),dtype=np.float)
Y_train = np.zeros((np.shape(A_image)[0] +np.shape(B_image)[0] +np.shape(C_image)[0] 
                   +np.shape(F_image)[0] +np.shape(P_image)[0] +np.shape(V_image)[0]),dtype=np.float)


j =0;
i=0 ;
X_train[j:np.shape(A_image)[0],:,:,:] = A_image ;
Y_train[j:np.shape(A_image)[0]] = 1
j = j+ np.shape(A_image)[0] ;
X_train[j:j+np.shape(B_image)[0],:,:,:] = B_image ;
Y_train[j:j+np.shape(B_image)[0]] = 2
j = j+ np.shape(B_image)[0] ;
X_train[j:j+np.shape(C_image)[0],:,:,:] = C_image ;
Y_train[j:j+np.shape(C_image)[0]] = 3
j = j+ np.shape(C_image)[0] ;
X_train[j:j+np.shape(F_image)[0],:,:,:] = F_image ;
Y_train[j:j+np.shape(F_image)[0]] = 4
j = j+ np.shape(F_image)[0] ;
X_train[j:j+np.shape(P_image)[0],:,:,:] = P_image ;
Y_train[j:j+np.shape(P_image)[0]] = 5
j = j+ np.shape(P_image)[0] ;
X_train[j:j+np.shape(V_image)[0],:,:,:] = V_image ;
Y_train[j:j+np.shape(V_image)[0]] = 6
j = j+ np.shape(V_image)[0] ;



            ############ Code to convert test image into matrix ###############
            
a_img_list = os.listdir('/home/bhumihar/Programming/Python/opencv/sample/Marcel-Test/A/uniform')
b_img_list = os.listdir('/home/bhumihar/Programming/Python/opencv/sample/Marcel-Test/B/uniform')
c_img_list = os.listdir('/home/bhumihar/Programming/Python/opencv/sample/Marcel-Test/C/uniform')
f_img_list = os.listdir('/home/bhumihar/Programming/Python/opencv/sample/Marcel-Test/Five/uniform')
p_img_list = os.listdir('/home/bhumihar/Programming/Python/opencv/sample/Marcel-Test/Point/uniform')
v_img_list = os.listdir('/home/bhumihar/Programming/Python/opencv/sample/Marcel-Test/V/uniform')

A_image = np.zeros((np.shape(a_img_list)[0],50,50,3),dtype=np.float)
A_value = np.zeros((np.shape(a_img_list)[0]),dtype=np.float)
i =0 ;
for img in a_img_list :
    file_name = "/home/bhumihar/Programming/Python/opencv/sample/Marcel-Test/A/uniform/"+img ;
    image = cv2.imread(file_name)
    if(image.shape!=(50,50,3)) :
        image = cv2.resize(image, (50, 50))
    A_image[i,:,:,:] = image
    A_value[i] = 1;
    i = i+1
    
B_image = np.zeros((np.shape(b_img_list)[0],50,50,3),dtype=np.float)
B_value = np.zeros((np.shape(b_img_list)[0]),dtype=np.float)
i =0 ;
for img in b_img_list :
    file_name = "/home/bhumihar/Programming/Python/opencv/sample/Marcel-Test/B/uniform/"+img ;
    image = cv2.imread(file_name)
    if(image.shape!=(50,50,3)) :
        image = cv2.resize(image, (50, 50))
    B_image[i,:,:,:] = image
    B_value[i] = 2;
    i = i+1
    
C_image = np.zeros((np.shape(c_img_list)[0],50,50,3),dtype=np.float)
C_value = np.zeros((np.shape(c_img_list)[0]),dtype=np.float)
i =0 ;
for img in c_img_list :
    file_name = "/home/bhumihar/Programming/Python/opencv/sample/Marcel-Test/C/uniform/"+img ;
    image = cv2.imread(file_name)
    if(image.shape!=(50,50,3)) :
        image = cv2.resize(image, (50, 50))
    C_image[i,:,:,:] = image
    C_value[i] = 3;
    i = i+1
    
F_image = np.zeros((np.shape(f_img_list)[0],50,50,3),dtype=np.float)
F_value = np.zeros((np.shape(f_img_list)[0]),dtype=np.float)
i =0 ;
for img in f_img_list :
    file_name = "/home/bhumihar/Programming/Python/opencv/sample/Marcel-Test/Five/uniform/"+img ;
    image = cv2.imread(file_name)
    if(image.shape!=(50,50,3)) :
        image = cv2.resize(image, (50, 50))
    F_image[i,:,:,:] = image
    F_value[i] = 4;
    i = i+1

P_image = np.zeros((np.shape(p_img_list)[0],50,50,3),dtype=np.float)
P_value = np.zeros((np.shape(p_img_list)[0]),dtype=np.float)
i =0 ;
for img in p_img_list :
    file_name = "/home/bhumihar/Programming/Python/opencv/sample/Marcel-Test/Point/uniform/"+img ;
    image = cv2.imread(file_name)
    if(image.shape!=(50,50,3)) :
        image = cv2.resize(image, (50, 50))
    P_image[i,:,:,:] = image
    P_value[i] = 5;
    i = i+1
    
V_image = np.zeros((np.shape(v_img_list)[0],50,50,3),dtype=np.float)
V_value = np.zeros((np.shape(v_img_list)[0]),dtype=np.float)
i =0 ;
for img in v_img_list :
    file_name = "/home/bhumihar/Programming/Python/opencv/sample/Marcel-Test/V/uniform/"+img ;
    image = cv2.imread(file_name)
    if(image.shape!=(50,50,3)) :
        image = cv2.resize(image, (50, 50))
    V_image[i,:,:,:] = image
    V_value[i] = 6;
    i = i+1

#print(np.shape(A_image))
#print(np.shape(B_image))
#print(np.shape(C_image))
#print(np.shape(F_image))
#print(np.shape(P_image))
#print(np.shape(V_image))

X_test = np.zeros((np.shape(A_image)[0] +np.shape(B_image)[0] +np.shape(C_image)[0] 
                   +np.shape(F_image)[0] +np.shape(P_image)[0] +np.shape(V_image)[0],50,50,3),dtype=np.float)
Y_test = np.zeros((np.shape(A_image)[0] +np.shape(B_image)[0] +np.shape(C_image)[0] 
                   +np.shape(F_image)[0] +np.shape(P_image)[0] +np.shape(V_image)[0]),dtype=np.float)

j =0;
i=0 ;
X_test[j:np.shape(A_image)[0],:,:,:] = A_image ;
Y_test[j:np.shape(A_image)[0]] = 1
j = j+ np.shape(A_image)[0] ;
X_test[j:j+np.shape(B_image)[0],:,:,:] = B_image ;
Y_test[j:j+np.shape(B_image)[0]] = 2
j = j+ np.shape(B_image)[0] ;
X_test[j:j+np.shape(C_image)[0],:,:,:] = C_image ;
Y_test[j:j+np.shape(C_image)[0]] = 3
j = j+ np.shape(C_image)[0] ;
X_test[j:j+np.shape(F_image)[0],:,:,:] = F_image ;
Y_test[j:j+np.shape(F_image)[0]] = 4
j = j+ np.shape(F_image)[0] ;
X_test[j:j+np.shape(P_image)[0],:,:,:] = P_image ;
Y_test[j:j+np.shape(P_image)[0]] = 5
j = j+ np.shape(P_image)[0] ;
X_test[j:j+np.shape(V_image)[0],:,:,:] = V_image ;
Y_test[j:j+np.shape(V_image)[0]] = 6
j = j+ np.shape(V_image)[0] ;


print('Train Image Dataset Matrix Size')
print(np.shape(X_train))
print(np.shape(Y_train))

print('Test Image Dataset Matrix Size')
print(np.shape(X_test))
print(np.shape(Y_test))


pyplot.imshow(toimage(X_train[0]))
pyplot.show()
print(Y_train[0])

      ######## Code to write train data into  file ######
file = open("gesture_train.npy", "wb")
np.savez(file, X_train=X_train,Y_train=Y_train)
file.seek(0)
npzfile = np.load("gesture_train.npy")
npzfile.files

      ######## Code to write train data into  file ######
file = open("gesture_test.npy", "wb")
np.savez(file, X_test=X_test,Y_test=Y_test)
file.seek(0)
npzfile = np.load("gesture_test.npy")
npzfile.files