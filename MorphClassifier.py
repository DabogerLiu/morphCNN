import os
from os import listdir
import numpy as np
import pickle
# import matplotlib.pyplot as plt
from skimage import transform, color
import keras
import random
import tensorflow as tf
from skimage import measure
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from keras.models import load_model
from skimage.filters import threshold_otsu
import glob
from morph_layers2D import Dilation2D, Erosion2D
from keras.utils import CustomObjectScope
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import numpy as np 
import pandas as pd 
from morph import dilate, erode, adapt
from utils import *

TRAIN_DIR = '/home/s/sl638/chest_xray/train'
VAL_DIR = '/home/s/sl638/chest_xray/val'
TEST_DIR = '/home/s/sl638/chest_xray/test'

from keras.layers import Input, Conv2D, concatenate, add, Dense, Dropout, MaxPooling2D, Flatten, \
                          UpSampling2D, Reshape, BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D, ZeroPadding2D, Activation,  AveragePooling2D, subtract
from keras.models import Model
import keras.backend as K
from keras.utils import multi_gpu_model

def lenet(in_image=(64, 64, 3)):
    img_in = Input(shape = in_image, name='image_in')
   # img_in_B = BatchNormalization(name='in_BN')(img_in)
    
    c01 = img_in
    c02 = img_in
#    c0 = concatenate([c01,c02], axis = 3)
#    c1 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(img_in)
#    c1 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(c1)
#    #c1 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(c1)
#    
#    c2 = subtract([c0,c1])
    
    c3 =  Dilation2D(6, (6,6),padding="same",strides=(1,1))(img_in)
    
    c4 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(img_in)
    c4 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(c4)
    
    c = concatenate([c3,c4], axis = 3)
    
    conv1 = Conv2D(32, 3, activation = 'relu', name='conv1_1')(c)
    pool1 = MaxPooling2D(pool_size=(2, 2),name='down1')(conv1)
    
    conv2 = Conv2D(32, 3, activation = 'relu',name='conv2_1' )(pool1)
   # pool2 = MaxPooling2D(pool_size=(2, 2),name='down2')(conv2)
    #down_4_f = Flatten(name='down_2_flat')(pool2)

    #down_classsify = Dense(128,activation='relu',name='classify_1')(down_4_f)
    #classification = Dense(1,activation='sigmoid',name='classification')(down_classsify)
    model = Model(inputs = img_in, outputs = classification)
    model.summary()

    return model
def adaptiveMNN(in_image=(64, 64, 3)): 
    img_in = Input(shape = (64, 64, 3), name='image_in')
    x1 = adapt(filters=6, kernel_size=(3,3),strides=(1,1),operation='m')(img_in)
    x1 = Activation('relu')(x1)
    x1 = adapt(filters=6, kernel_size=(3,3),strides=(1,1),operation='m')(x1)
    x1 = Activation('relu')(x1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name='down1')(x1)
    x2 = adapt(filters=12, kernel_size=(3,3),strides=(1,1),operation='m')(pool1)
    x2 = Activation('relu')(x2)
    x2 = adapt(filters=12, kernel_size=(3,3),strides=(1,1),operation='m')(x2)
    x2 = Activation('relu')(x2)
    pool3 = MaxPooling2D(pool_size=(2, 2),name='down3')(x2)
    x4 = Flatten()(pool3)
    x5 = Dense(units=128, activation='relu')(x4)
    output = Dense(units=1, activation='sigmoid')(x6)
    
    model = Model(inputs = img_in, outputs = output)
    model.summary()
    return model
    
def adaptiveMCNN(in_image=(64, 64, 3)): 
    img_in = Input(shape = (64, 64, 3), name='image_in')
    x1 = adapt(filters=6, kernel_size=(3,3),strides=(1,1),operation='m')(img_in)
    x1 = Activation('relu')(x1)
    x1 = adapt(filters=6, kernel_size=(3,3),strides=(1,1),operation='m')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(32, 3, activation = 'relu', name='conv1_1')(x1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name='down1')(x1)
    x2 = Conv2D(32, 3, activation = 'relu', name='conv2_1')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2),name='down3')(x2)
    x4 = Flatten()(pool2)
    x5 = Dense(units=128, activation='relu')(x4)
    output = Dense(units=1, activation='sigmoid')(x5)
    model = Model(inputs = img_in, outputs = output)
    model.summary()
    return model
    
def mnn(in_image=(64, 64, 3)):
    img_in = Input(shape = in_image, name='image_in')
    c01 =  Dilation2D(3, (3,3),padding="same",strides=(1,1))(img_in)
    c02 = Erosion2D(3, (3,3),padding="same",strides=(1,1))(img_in)
    c1 = concatenate([c01,c02], axis = 3)
    #c1 = Conv2D(32, 3, activation = 'relu', name='conv1_1')(c1)
    #pool1 = MaxPooling2D(pool_size=(2, 2),name='down1')(c1)
    
    c11 =  Dilation2D(4, (4,4),padding="same",strides=(1,1))(c1)
    c12 = Erosion2D(4, (4,4),padding="same",strides=(1,1))(c1)
    c2 = concatenate([c11,c12], axis = 3)
    #c2 = Conv2D(32, 3, activation = 'relu')(c2)
    pool2 = MaxPooling2D(pool_size=(2, 2),name='down2')(c2)
    
    c21 =  Dilation2D(3, (4,4),padding="same",strides=(1,1))(pool2)
    c22 = Erosion2D(3, (4,4),padding="same",strides=(1,1))(pool2)
    c3 = concatenate([c21,c22], axis = 3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2),name='down3')(c3)
    
    down_4_f = Flatten(name='down_2_flat')(pool3)

    down_classsify = Dense(128,activation='relu',name='classify_1')(down_4_f)
    classification = Dense(1,activation='sigmoid',name='classification')(down_classsify)
    model = Model(inputs = img_in, outputs = classification)
    model.summary()

    return model
    
from keras import backend as K

def F1(y_true, y_pred):
    
    def precision(y_true, y_pred):
        """ Batch-wise average precision calculation

        Calculated as tp / (tp + fp), i.e. how many selected items are relevant
        Added epsilon to avoid the Division by 0 exception
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    def recall(y_true, y_pred):
        """ Batch-wise average recall calculation

        Computes the Recall metric, or Sensitivity or True Positive Rate  
        Calculates as tp / (tp + fn) i.e. how many relevant items are selected

        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
   
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*(precision*recall)/(precision+recall+K.epsilon())
    

def compile_model():

    #model = lenet()
    #model = mnn()
    model = adaptiveMNN()
    #model = adaptiveMCNN()
    model =  multi_gpu_model(model,gpus=4)
    #opti1 = keras.optimizers.Adam(lr=0.01)
    
    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = [F1])

    return model

def train(epoch=25):
    model = compile_model()
    batch_size = 32
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory(TRAIN_DIR,
                                                 target_size = (64, 64),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')
    valid_set = test_datagen.flow_from_directory(VAL_DIR,
                                            target_size = (64, 64),
                                            batch_size = batch_size,
                                            class_mode = 'binary')
    n_training_files = len(training_set.filenames)
    n_valid_files = len(valid_set.filenames)

    history = model.fit_generator(training_set,
                         steps_per_epoch = n_training_files/batch_size,
                         epochs = epoch,
                         validation_data = valid_set,
                         validation_steps = n_valid_files/batch_size)
                         
    test_set = test_datagen.flow_from_directory(TEST_DIR,
                                            target_size = (64, 64),
                                            batch_size = batch_size,
                                            class_mode = 'binary')

    test_accuracy = model.evaluate_generator(test_set,steps=624)
    print('The testing accuracy is :', test_accuracy[1]*100, '%')
    
if __name__ == "__main__":

    print("===============")
    train(epoch=50)

    #test_accuracy = model.evaluate_generator(test_set,steps=624)
    #print('The testing accuracy is :', test_accuracy[1]*100, '%')