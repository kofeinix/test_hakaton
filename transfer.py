# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 00:30:02 2022

@author: Юрий
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 00:11:57 2022

@author: 19761378
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
'''for dirname, _, filenames in os.walk('/kaggle/input'):
for filename in filenames:
print(os.path.join(dirname, filename))'''

import cv2

from matplotlib import pyplot as plt

import tensorflow as tf
from keras.utils import np_utils
# for cnn model
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import Dense, Conv2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Reshape, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

data = pd.read_csv("fer2013.csv")
data.head(10)

train_set = data[(data.Usage == 'Training')]
test_set = data[(data.Usage == 'PrivateTest')]

X_train = np.array(list(map(str.split, train_set.pixels)), np.float32)
X_test = np.array(list(map(str.split, test_set.pixels)), np.float32)
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)


X_train=X_train/255
X_test=X_test/255

y_train = train_set.emotion
y_test = test_set.emotion

y_train = np_utils.to_categorical(y_train, 7)
y_test = np_utils.to_categorical(y_test, 7)


base_model= keras.applications.Xception(weights = 'imagenet', input_shape=(150,150,3), include_top=False)
base_model.trainable = False



model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7 , activation='softmax'))

print(model.summary())

optimizer=Adam(learning_rate=0.001)

model.compile(loss='categor_crossentropy',
optimizer=optimizer ,
metrics=['accuracy'])


history = model.fit(X_train, y_train, batch_size=128, validation_data=(X_train,y_train), epochs = 50)

print("Accuracy of our model on validation data : " , model.evaluate(X_train,y_train)[1]*100 , "%")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
    model.save_weights("fer.h5")
