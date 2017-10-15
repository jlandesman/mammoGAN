import numpy as np

## Use GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

np.random.seed(123)

from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
import pandas as pd
import cv2

## Load in data
TRAIN_DIR = "data/mias_mini-train"
TEST_DIR = "data/mias_mini-test"
images = []
for imgName in os.listdir(TRAIN_DIR):
    if "png" in imgName:
        images.append(cv2.imread(TRAIN_DIR + '/' + imgName))
        
X_train = np.array(images[1:])  

images = []
for imgName in os.listdir(TEST_DIR):
    if "png" in imgName:
        images.append(cv2.imread(TEST_DIR + '/' + imgName))
        
X_test = np.array(images[1:])

## Load in labels
train_labels = pd.read_csv(os.path.join(TRAIN_DIR,"train_labels.csv"))
test_labels = pd.read_csv(os.path.join(TEST_DIR,"test_labels.csv"))
Y_train = np_utils.to_categorical(train_labels, 3)
Y_test = np_utils.to_categorical(test_labels, 3)

model = VGG16(weights=None, input_shape=(1024,1024,3), classes=3)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, 
          batch_size=1, epochs=1, verbose=1)

model.save('/home/jlandesman/models/vgg.h5')

print ('Test loss:', score[0])
print ('Test accuracy:', score[1])
