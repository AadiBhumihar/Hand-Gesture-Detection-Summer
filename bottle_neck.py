#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bhumihar
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils import np_utils

# dimensions of our images.
img_width, img_height = 150, 150

datagen = ImageDataGenerator(rescale=1. / 255)

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '/home/bhumihar/Programming/Python/opencv/sample/project/Marcel-train'  
validation_data_dir = '/home/bhumihar/Programming/Python/opencv/sample/project/Marcel-test'
nb_train_samples = 4872
nb_validation_samples = 377
epochs = 50
batch_size = 29

    # build the VGG16 network
def bottleneck_feature() :
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False) 
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size,verbose=1)
    
    file = open('bottleneck_features_train.npy', 'wb')
    np.savez(file, bottleneck_features_train)
    
    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples//batch_size,verbose=1)
    file = open('bottleneck_features_validation.npy', 'wb')
    np.savez(file, bottleneck_features_validation)
    
def train_top_model() :
    
    npyfile = np.load("bottleneck_features_train.npy")
    a_size ,b_size,c_size,f_size,p_size,v_size = 1329,487,572,654,1395,435
    ac_size ,bc_size,cc_size,fc_size,pc_size,vc_size = 58,60,64,75,64,56
    train_data = npyfile['arr_0'] 
    train_labels =np.array(([0] * int(a_size)) + ([1] * int(b_size)) + ([2] * int(c_size))
                          + ([3] * int(f_size))+ ([4] * int(p_size))+ ([5] * int(v_size)))
    
    npyfile = np.load("bottleneck_features_validation.npy")
    validation_data = npyfile['arr_0'] 
    validation_labels = np.array(([0] * int(ac_size)) + ([1] * int(bc_size)) + ([2] * int(cc_size))
                                + ([3] * int(fc_size))+ ([4] * int(pc_size))+ ([5] * int(vc_size)))
    
    
    train_labels = np_utils.to_categorical(train_labels, 6)
    validation_labels = np_utils.to_categorical(validation_labels, 6)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    
bottleneck_feature() 
train_top_model() 