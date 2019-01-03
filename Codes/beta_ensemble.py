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

dataset = np.loadtxt('outfile1.csv', delimiter=",",skiprows=1)
size=dataset.shape[0]
feature_size=dataset.shape[1]-1
classes = np.unique(dataset[:,-1])
num_classes=17 #classes.size
start=int(0.5*size)
end = int(0.8*size)
train1_x , train1_y = dataset[start:end,0:-1] , dataset[start:end,-1]
train1_y = to_categorical(train1_y)  #converts to one hot

dataset = np.loadtxt('outfile2.csv', delimiter=",",skiprows=1)
train_size=int(0.8*size)
train2_x , train2_y = dataset[start:end,0:-1] , dataset[start:end,-1]
train2_y = to_categorical(train2_y)  #converts to one hot

dataset = np.loadtxt('outfile3.csv', delimiter=",",skiprows=1)
train_size=int(0.8*size)
train3_x , train3_y = dataset[start:end,0:-1] , dataset[start:end,-1]
train3_y = to_categorical(train3_y)  #converts to one hot


classes = np.unique(train1_y)
num_classes=classes.size
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
print(X.shape[1])
print('\n \n')
print(Y.shape[1])

model = Sequential()
model.add(Dense(100, input_dim=51, activation='relu'))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(Dense(100, activation='relu'))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(17, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# Fit the model
model.fit(X, Y, epochs=30, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save('model_ensemble.h5')


