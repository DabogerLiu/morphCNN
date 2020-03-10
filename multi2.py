import os
from skimage.transform import resize
import numpy as np
from morph import dilate, erode, adapt
import keras
from keras.datasets import mnist
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, Activation, Add
from keras import optimizers
from keras.models import Model
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import skimage.io as io
from grayscale_morph import gray_dilate, gray_erode, sliding_window
import argparse
from utils import *
from keras import activations

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--gpu_id', type=int, default=3)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import cv2

#model
inputs = Input(shape=(28, 28, 1))
#x1 = dilate(filters=1, kernel_size=(3,3),strides=(1,1),operation='m')(inputs)
x1 = adapt(filters=1, kernel_size=(3,3),strides=(1,1),operation='m')(inputs)
#x1 = Add()([inputs, x1])
#x1 = activations.relu(x1)
x2 = adapt(filters=1, kernel_size=(3,3), strides=(1,1), operation='m')(x1)
#x2 = Add()([x1, x2])
x3 = adapt(filters=1, kernel_size=(3,3), strides=(1,1), operation='m')(x2)
#x2 = activations.tanh(x2)
#x1 = keras.layers.Conv2D(filters=1,kernel_size=(3,3), strides=(1,1), padding='SAME')(inputs)
#x2 = keras.layers.Conv2D(filters=1,kernel_size=(3,3), strides=(1,1), padding='SAME')(x1)
#x3 = keras.layers.Conv2D(filters=1,kernel_size=(3,3), strides=(1,1), padding='SAME')(x2)
#x2 = keras.activations.softmax(x1, axis=-1)
#x2 = dilate(filters=1, kernel_size=(3,3), strides=(1,1),operation='m')(x1)
#x3 = dilate(filters=1, kernel_size=(3,3),strides=(1,1),operation='m')(x2)
#x4 = erode(filters=1, kernel_size=(3,3),strides=(1,1),operation='m')(x3)
M = Model(inputs=inputs,outputs=x3)

M.summary()

optimizer = optimizers.SGD(lr=args.lr, momentum=0.0, decay=args.lr/args.epochs,nesterov=False)
M.compile(loss='mse', optimizer=optimizer)

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255.
x_train = x_train[:20000:,:,:,:]


y_train, kernel = get_y(x_train)
print(kernel)
y_test = get_y(x_test)



M.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,verbose=1)

weights = M.get_weights()
kernel = np.ones((3,3),np.uint8)
print(kernel)
print("whoops!")
predict1 = np.reshape(M.get_weights()[0], (3,3))
predict2 = np.reshape(M.get_weights()[3], (3,3))
print('adapt weight 1:', indicator(weights[2]))
print('adapt weight 2:', indicator(weights[5]))
print('adapt weight 3:', indicator(weights[8]))
#print(np.sum(abs(predict-kernel1)))
print('predict weight 1:', predict1)
print('predict weight 2:', predict2)

kernel = np.ones((3,3),np.uint8)

sum_dis = 0
for i in range(100):
    M.fit(x_train, y_train, batch_size = 64, epochs=100, verbose=1)
    predict = np.reshape(np.round(M.get_weights()[0]), (3,3)).astype(int)
    if np.array_equal(predict, kernel1):
        sum_dis += 1

print(sum_dis)

similarity = sum_dis/100
print(sum_dis)
print(similarity)

M.fit(x_train, y_train, batch_size = 64, epochs=20, verbose=1)
predict = np.reshape(M.get_weights()[0], (3,3))
print('predict SE:' ,predict)
print('target SE:' , kernel1)
if np.array_equal(predict, kernel1):
    print('yes!')
print('bias:' ,M.get_weights()[1])
np.save('bias.npy', M.get_weights()[1])
origin = x_train[1]
target = y_train[1]
prediction = M.predict(np.reshape(origin, [1,28,28,1]))
print('input image:', np.reshape(origin, [28,28]))
print('target image:' ,np.reshape(origin, [28,28]))
print('predicted image:', np.reshape(origin, [28,28]))
np.save('input_image.npy', np.reshape(origin, [28, 28]))
np.save('target_image.npy', np.reshape(target, [28, 28]))
np.save('predicted_image.npy', np.reshape(prediction, [28, 28]))



samples = 4

#origin = io.imread('Lena.jpg', as_grey=True)
origin = resize(origin, (28, 28))
kernel = np.ones((3,3),np.uint8)
#kernel[0,0]=1
#kernel[1,1]=1
#kernel[2,2]=1
print(kernel)
target = cv2.dilate(origin,kernel,iterations=1)
print(origin.shape)

numbers = np.random.randint(low=0, high=5000, size=samples)
plt.figure()
for i in range(samples):
    index = i + 1

    origin = x_train[numbers[i]]
    target = y_train[numbers[i]]

    plt.subplot(3, samples, index + samples * 0)
    image = np.reshape(origin, [28, 28])
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('original')

    plt.subplot(3, samples, index + samples * 1)
    image = np.reshape(target, [28, 28])
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('target')

    plt.subplot(3, samples, index + samples * 2)
    prediction = M.predict(np.reshape(origin, [1,28, 28,1]))
    image = np.reshape(prediction, [28, 28])
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    #plt.title('prediction')

plt.tight_layout()
plt.show()
plt.savefig('dilation_gray.png')
