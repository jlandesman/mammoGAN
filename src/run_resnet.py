import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
#import seaborn as sns
#from IPython.display import display 
from PIL import Image
from keras.applications.resnet50 import ResNet50 
import time
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.models import load_model


from keras.callbacks import Callback
import warnings
import numpy as np

class ModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        num_outputs = len(self.model.outputs)
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.layers[-(num_outputs+1)].save_weights(filepath, overwrite=True)
                        else:
                            self.model.layers[-(num_outputs+1)].save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.layers[-(num_outputs+1)].save_weights(filepath, overwrite=True)
                else:
                    self.model.layers[-(num_outputs+1)].save(filepath, overwrite=True)

TRAIN_DIR = "/mnt/disks/patches/calcifications/train/"
TEST_DIR = "/mnt/disks/patches/calcifications/test/"
IM_WIDTH, IM_HEIGHT = 256, 256
FC_SIZE = 256
batch_size = 120
NUM_CLASSES = 4
NUM_EPOCHS = 50


train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
)


import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

model = Sequential()
model.add(ResNet50(include_top = False, 
                   input_shape = (256, 256, 3), classes = NUM_CLASSES))

## Add in last 3 layers
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.load_weights('/home/jlandesman/model_history/weights-improvement-03-0.52.hdf5')

filepath="/home/jlandesman/model_history/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

## Parallelize to attempt multi-GPU effort
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(optimizer= Adam(lr=0.0002,beta_1=0.9, beta_2=0.999), 
                                       loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer= Adam(lr=0.002, beta_1=0.9, beta_2=0.999), 
#                                       loss='categorical_crossentropy', metrics=['accuracy'])


start = time.time()
modelFit = parallel_model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples/batch_size,
            epochs=NUM_EPOCHS,
            verbose = 1,
            validation_data=test_generator,
            validation_steps=300, 
            callbacks=callbacks_list)
end = time.time()

total_time = int(end-start)
time_per_epoch = total_time/NUM_EPOCHS

forecasted_time = 100000 * time_per_epoch / (train_generator.samples + test_generator.samples)
print ()
print ()
print ("Model took " + str(total_time) + " seconds to run" )
print ("Model takes " + str(time_per_epoch) + " seconds to run")
print ("Approximate time taken per epoch for 100,000 images is " + str(forecasted_time) + " seconds")

