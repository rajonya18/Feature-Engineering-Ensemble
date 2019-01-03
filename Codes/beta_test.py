import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import model_from_json
import numpy as np
import os
from tensorflow.python.keras.models import load_model

num_classif=3
num_classes=17

dataset = np.loadtxt('outfile1.csv', delimiter=",",skiprows=1)
size=dataset.shape[0]
feature_size=dataset.shape[1]-1
classes = np.unique(dataset[:,-1])
num_classes=17 #classes.size
train_size=int(0.8*size)
train1_x , train1_y = dataset[int(0.8*size):size,0:-1] , dataset[int(0.8*size):size,-1]
train1_y = to_categorical(train1_y)  #converts to one hot

dataset = np.loadtxt('outfile2.csv', delimiter=",",skiprows=1)
train_size=int(0.8*size)
train2_x , train2_y = dataset[int(0.8*size):size,0:-1] , dataset[int(0.8*size):size,-1]
train2_y = to_categorical(train2_y)  #converts to one hot

dataset = np.loadtxt('outfile3.csv', delimiter=",",skiprows=1)
train_size=int(0.8*size)
train3_x , train3_y = dataset[int(0.8*size):size,0:-1] , dataset[int(0.8*size):size,-1]
train3_y = to_categorical(train3_y)  #converts to one hot


feature_size=num_classes*num_classif

model1 = load_model('model1.h5')
model2 = load_model('model2.h5')
model3 = load_model('model3.h5')

score = model1.evaluate(train1_x, train1_y, verbose=0)
print("model1====== %s: %.2f%%" % (model1.metrics_names[1], score[1]*100))
score = model2.evaluate(train2_x, train2_y, verbose=0)
print("model2====== %s: %.2f%%" % (model1.metrics_names[1], score[1]*100))
score = model3.evaluate(train3_x, train3_y, verbose=0)
print("model3====== %s: %.2f%%" % (model1.metrics_names[1], score[1]*100))

print('\n')

predictions1 =model1.predict(train1_x)
predictions2 =model2.predict(train2_x)
predictions3 =model3.predict(train3_x)

X=np.concatenate((predictions1,predictions2,predictions3),axis=1)
Y=train3_y

model_ensemble=load_model('model_ensemble.h5')
score = model_ensemble.evaluate(X,Y, verbose=0)
print("model_ensemble====== %s: %.2f%%" % (model_ensemble.metrics_names[1], score[1]*100))
