import os
from os import listdir
import numpy as np
import pickle
# import matplotlib.pyplot as plt
from skimage import transform, color
import keras
import random
import tensorflow as tf
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from skimage import measure
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from keras.models import load_model
from skimage.filters import threshold_otsu
import glob
#from morph_layers2D import Dilation2D, Erosion2D
from keras.utils import CustomObjectScope
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import cv2

# 1. pre-defined variables
batch_size = 32           # batch size
height, width = 512,512  # input size
n_class = 3               # number of class

path_here = '/home/s/sl638/datasets/ultrosound/'
folders = listdir(path_here)
#print(folders)
file_folder = []
for filename in glob.iglob(path_here + '**/*.png', recursive=True):
     file_folder.append(filename)
     #print(filename)
 
#print(file_folder)
random.shuffle(file_folder)
train_filenames = file_folder[0:590]
valid_filenames = file_folder[590:]
data_len = len(train_filenames)
test_data_len = len(valid_filenames)
#print(train_data_len)
#print(test_data_len)


def get_batch(folder,batch_size):
    while True:
        c = np.random.choice(folder, batch_size*4)
        
        count_batch = 0
        
        img_in_all = []
        img_target_all = []
        for each_file in c:
            img = cv2.imread(each_file)
            img_in =  np.array(img)
            img_target =int(each_file.split(os.path.sep)[-2])  
            img_target = np.int32(img_target)

            

            img_in = transform.resize(img_in, (height, width, 1), mode='reflect')
            img_target = keras.utils.to_categorical(img_target, 3)

            img_in_all.append(img_in)
            img_target_all.append(img_target)
            
            if count_batch >= batch_size-1:
                break
            count_batch += 1
            
        img_in_all = np.array(img_in_all)
        img_in_all = np.reshape(img_in_all, [batch_size, height, width, 1])
        img_target_all = np.array(img_target_all)
        img_target_all = np.reshape(img_target_all, [batch_size, n_class])

        yield ({'image_in': img_in_all}, \
              {'classification': img_target_all})

from keras.layers import Input, Conv2D, concatenate, add, Dense, Dropout, MaxPooling2D, Flatten, \
                          UpSampling2D, Reshape, BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D, ZeroPadding2D, Activation, AveragePooling2D
from keras.models import Model
import keras.backend as K
from keras.utils import multi_gpu_model

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    
    #bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization( name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization( name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    #bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def Res50(in_image=(height, width, 1)):
    img_in = Input(shape = in_image, name='image_in')
    
    
    #bn_axis = 3
    x = ZeroPadding2D((3, 3))(img_in)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    classification = Dense(3,activation='relu',name='classification')(x)
    
    model = Model(img_in,  classification)
    model.summary()
    return model


def vgg16(in_image=(height, width, 1)):
    img_in = Input(shape = in_image, name='image_in')
   # img_in_B = BatchNormalization(name='in_BN')(img_in)
    
    #c0 = img_in_B
    #c0 = Conv2D(12, 8, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv0_0')(c0)
    
    #c1 = Erosion2D(6, (6,6),padding="same",strides=(1,1))(img_in)
    
    #c1 = Dilation2D(6, (6,6),padding="same",strides=(1,1))(c1)
    
    #c12= concatenate([c1,c2], axis = 3) #10 filters

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_1')(img_in)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_2')(conv1)
    #conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv1_3')(conv1)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name='down1')(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_1' )(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_2')(conv2)
    #conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_3')(conv2)
    drop2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),name='down2')(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_1')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_2')(conv3)
    #conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_3')(conv3)
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),name='down3')(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4_1')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv4_2')(conv4)
    #conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv4_3')(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),name='down4')(drop4)

    down_4_f = Flatten(name='down_2_flat')(pool4)

    down_classsify = Dense(512,activation='relu',name='classify_1')(down_4_f)
    down_classsify = Dropout(0.4)(down_classsify)
    down_classsify = Dense(128,activation='relu',name='classify_2')(down_classsify)
    down_classsify = Dropout(0.4)(down_classsify)
    classification = Dense(3,activation='relu',name='classification')(down_classsify)
    model = Model(inputs = img_in, outputs = classification)
    model.summary()

    return model
    
import keras.backend as K

def compile_model():

    #model = vgg16()
    model = Res50()
    model =  multi_gpu_model(model,gpus=4)
    opti1 = keras.optimizers.Adam(lr=0.0005)
    
    model.compile(optimizer=opti1,
                  loss=['categorical_crossentropy'],
                  metrics = {'classification':'accuracy'})

    return model
    
def train(epoch=5):
    model = compile_model()

    history = model.fit_generator(get_batch(train_filenames,batch_size), validation_data = get_batch(valid_filenames,batch_size), \
                                 steps_per_epoch=int(data_len*10 / batch_size), epochs=epoch, validation_steps= int(len(valid_filenames)*10 / batch_size))
    model.save('/home/s/sl638/Ultrosound/CNN.h5')
    with open('/home/s/sl638/Ultrosound/history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

from matplotlib import pyplot as plt
def plot_loss():
    with open('/home/s/sl638/Ultrosound/history.pkl', 'rb') as file_pi:
        history = pickle.load(file_pi)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history["classification_loss"], label="classification_loss")
    plt.plot(history["val_classification_loss"], label="val_classification_loss")
    plt.legend()
    plt.subplot(122)
    plt.plot(history["classification_acc"],label = "classification_acc")
    plt.plot(history["val_classification_acc"],label = "val_classification_acc")
    plt.legend()
    plt.show()
    plt.savefig('/home/s/sl638/Ultrosound/train_CNN.png')


if __name__ == "__main__":

    print("===============")
    train(epoch=100)
    plot_loss()