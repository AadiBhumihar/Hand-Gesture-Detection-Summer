#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: bhumihar
"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K




top_model_weights_path = 'bottleneck_fc_model.h5'

img_width, img_height = 150, 150

train_data_dir = '/home/bhumihar/Programming/Python/opencv/sample/project/Marcel-train'  
validation_data_dir = '/home/bhumihar/Programming/Python/opencv/sample/project/Marcel-test'
nb_train_samples = 4872
nb_validation_samples = 377
nb_epoch = 10
batch_size = 29

if K.image_data_format() == 'channels_first':
    inp_shape = (3, img_width, img_height)
else:
    inp_shape = (img_width, img_height, 3)

inp_tensor = Input(shape=inp_shape)
# build the VGG16 network
model = Sequential()
model.add(applications.VGG16(weights='imagenet', include_top=False,input_tensor=inp_tensor))
print('Model loaded.')


top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(6, activation='sigmoid'))


top_model.load_weights(top_model_weights_path)


model.add(top_model)


for layer in model.layers[:25]:
    layer.trainable = False


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)