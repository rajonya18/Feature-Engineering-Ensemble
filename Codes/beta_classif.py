import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import load_model
import numpy as np
import os

dataset = np.loadtxt('outfile1.csv', delimiter=",",skiprows=1)
size=dataset.shape[0]
feature_size=dataset.shape[1]-1
classes = np.unique(dataset[:,-1])
num_classes=classes.size
train_size=int(0.5*size)
train1_x , train1_y = dataset[0:train_size,0:-1] , dataset[0:train_size,-1]
train1_y = to_categorical(train1_y)  #converts to one hot

dataset = np.loadtxt('outfile2.csv', delimiter=",",skiprows=1)
train_size=int(0.5*size)
train2_x , train2_y = dataset[0:train_size,0:-1] , dataset[0:train_size,-1]
train2_y = to_categorical(train2_y)  #converts to one hot

dataset = np.loadtxt('outfile3.csv', delimiter=",",skiprows=1)
train_size=int(0.5*size)
train3_x , train3_y = dataset[0:train_size,0:-1] , dataset[0:train_size,-1]
train3_y = to_categorical(train3_y)  #converts to one hot


model = Sequential()
model.add(Dense(200, input_dim=feature_size, activation='relu'))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(Dense(150, activation='relu'))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(17, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Fit the model
model.fit(train1_x, train1_y, epochs=30, batch_size=100)
score = model.evaluate(train1_x, train1_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
model.save('model1.h5')  # creates a HDF5 file 'my_model.h5'

# Fit the model
model.fit(train2_x, train2_y, epochs=30, batch_size=100)
score = model.evaluate(train2_x, train2_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
model.save('model2.h5')  # creates a HDF5 file 'my_model.h5'


# Fit the model
model.fit(train3_x, train3_y, epochs=30, batch_size=100)
score = model.evaluate(train3_x, train3_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
model.save('model3.h5')  # creates a HDF5 file 'my_model.h5'



