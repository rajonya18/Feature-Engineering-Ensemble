import glob
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

num_classes = 2
dataset = 'Hill_Valley'

def predict(dataset_X,dataset_y, id):
	# Return the probability of each class
	# Load the model
	print('Loading model '+'model'+str(id)+'.h5')
	model = load_model('model'+str(id)+'.h5')
	
	dataset_y = to_categorical(dataset_y)
	score = model.evaluate(dataset_X,dataset_y, verbose=0)
	print("model"+str(id)+"====== %s: %.2f%%" % (model.metrics_names[1], score[1]*100))
	# Predict probability of each class
	predictions =model.predict(dataset_X)
	return predictions



def test(dataset_name):
	# Find the accuracy on the test dataset

	for i in range(3):
		dataset_X = pd.read_csv(dataset_name+'_test'+str(i+1)+'.csv')
		dataset_y = dataset_X['class']
		dataset_X.drop('class',axis=1,inplace=True)
		# Predict and concatenate
		if (i==0):
			predictions1=predictions2=predict(dataset_X,dataset_y, (i+1))
		else:
			predictions1+=predict(dataset_X,dataset_y, (i+1))
			predictions2*=predict(dataset_X,dataset_y, (i+1))

	y_pred1=np.argmax(predictions1,axis=1)
	y_pred2=np.argmax(predictions2,axis=1)
	print('sum '+str(accuracy_score(dataset_y,y_pred1)))
	print('product '+str(accuracy_score(dataset_y,y_pred2)))	

test('../Data/'+dataset)