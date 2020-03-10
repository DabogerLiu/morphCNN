import os
from skimage.transform import resize
import numpy as np
#from morph import dilate, erode, adapt
import keras
from keras.datasets import mnist
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, Activation, Add
from keras import optimizers
from keras.models import Model
import matplotlib.pyplot as plt
import cv2
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from matplotlib import pyplot

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs




def get_y(x):
    y = []
    kernel1=np.zeros((3,3), np.uint8)
    kernel1[1,0]=1
    kernel1[1,1]=1
    kernel1[1,2]=1
    kernel1[0,1]=1
    kernel1[2,1]=1

    kernel3 = np.zeros((3,3), np.uint8)
    kernel3[0,1]=1
    kernel3[1,1]=1
    kernel3[2,1]=1

    kernel2=np.zeros((3,3),np.uint8)*10
    kernel2[1,0]=1
    kernel2[1,1]=1
    kernel2[1,2]=1

    kernel2 = np.ones((5,5), np.uint8)
    kernel2[0,0]=0
    kernel2[0,1]=0
    kernel2[1,0]=0
    kernel2[0,3]=0
    kernel2[0,4]=0
    kernel2[1,4]=0
    #kernel1 = kernel2
    #kernel2[3,0]=kernel2[3,4]=0
    #kernel2[4,0]=kernel2[4,1]=kernel2[4,3]=kernel2[4,4]=0

    #kernel3 = np.zeros((5,5), np.uint8)
    #kernel3[0,4]=kernel3[1,3]=kernel3[2,2]=kernel3[3,1]=kernel3[4,0]=1
    #kernel3[0,0]=kernel3[1,1]=kernel3[3,3]=kernel3[4,4]=1

    #kernel1 = np.ones((1,5), np.uint8)

    #kernel3[1,1]=1
    #kernel3[2,1]=1
    #kernel1=np.random.random((3,3))
    #print(kernel1)
    #kernel2=np.random.random((3,3))
    #print(kernel2)
    #kernel3=np.random.random((3,3))
    #print(kernel3)
    #kernel4 = np.random.random((3,3))
    #print(kernel4)
    #print(kernel)
    #kernel = np.zeros((3,3), np.uint8)
    #kernel[0,0]=kernel[1,1]=kernel[2,2]=1
    #print(kernel2)
    #print(kernel2)
    #print(kernel3)
    #gaussian_kernel = cv2.getGaussianKernel(5,5)
    #print(gaussian_kernel)
    for imgno in range(x.shape[0]):
        #x1 = cv2.erode(x[imgno],kernel1, iterations=1)
        #x2 = cv2.dilate(x1,kernel2,iterations=1)
        #x3 = cv2.dilate(x2,kernel3,iterations=1)
        #y.append(cv2.morphologyEx(x[i, cv2.MORPH_OPEN, kernel2))
        #print(x[imgno][:, :, 0].shape)
        #y.append(gray_dilate(x[imgno][:,:,0], kernel1)[:, :, np.newaxis])
        x1 = cv2.erode(x[imgno], kernel1)
        x2 = cv2.erode(x1, kernel1)
        y.append(cv2.dilate(x2, kernel1))
        #x1 = cv2.filter2D(x[imgno], -1, kernel1)
        #x2 = cv2.filter2D(x1, -1, kernel2)
        #y.append(cv2.filter2D(x2, -1, kernel3))
    y = np.array(y)
    y = y.reshape(y.shape[0], 28, 28, 1)
    #y = y.astype('float32')/ 255.
    return y, kernel1

def indicator(x):
    result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    return result

def tanh(x):
    return (K.exp(x) - K.exp(-x)) / (K.exp(x) + K.exp(-x))

def ApproxSign(x):
    x = tf.reshape(x, [])
    #print(x)
    #result = tf.case([tf.less(x, tf.constant(-1, dtype=tf.float32)), tf.constant(-1, dtype=tf.float32)],
    #                 [(tf.less(x, tf.constant(0, dtype=tf.float32)), (2*x+x^2)), (tf.greater_equal(x, tf.constant(1, dtype=tf.float32)), tf.constant(1, dtype=tf.float32))],
    #     default=2*x-x^2, exclusive=True)
    #print(tf.less(x, 0))
    result = tf.cond(tf.less(x, tf.constant(0, dtype=tf.float32)), lambda: (2*x+x*x), lambda: (2*x-x*x))

    return result

def SoftSign(x):
    return x / (1+abs(x))


def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
