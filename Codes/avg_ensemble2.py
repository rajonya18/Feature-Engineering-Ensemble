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
dataset = 'horse-colic'

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

def get_wt(dataset_name):
	#************** PREDICTION *************************************
	wt=[0]*3
	for i in range(3):
		dataset_X = pd.read_csv(dataset_name+'_val'+str(i+1)+'.csv')
		dataset_y = dataset_X['class']
		dataset_X.drop('class', axis=1, inplace=True)
		dataset_y = to_categorical(dataset_y)

		model = load_model('model'+str(i+1)+'.h5')
		score = model.evaluate(dataset_X,dataset_y, verbose=0)
		wt[i]=score[1]
		print(wt[i])		

	return wt	

# def calc_majority_voting(dataset_name):

# 	pred = [0]*3
# 	for i in range(3):
# 		dataset_X = pd.read_csv(dataset_name+'_test'+str(i+1)+'.csv')
# 		dataset_y = dataset_X['class']
# 		dataset_X.drop('class',axis=1,inplace=True)
# 		# Predict and concatenate
# 		pred[i] = np.argmax( predict(dataset_X,dataset_y, (i+1)) , axis=1 )
	

def test(dataset_name):
	# Find the accuracy on the test dataset

	w = get_wt(dataset_name)
	for i in range(3):
		dataset_X = pd.read_csv(dataset_name+'_test'+str(i+1)+'.csv')
		dataset_y = dataset_X['class']
		dataset_X.drop('class',axis=1,inplace=True)
		# Predict and concatenate
		predictions = predict(dataset_X,dataset_y, (i+1))
		if (i==0):
			predictions1=predictions
			predictions2=predictions
			predictions3=predictions
			predictions4=w[i]*predictions
		else:
			predictions1=predictions1+predictions  							#sum rule
			predictions2=predictions2*predictions							#product rule	
			predictions3=np.maximum(predictions3,predictions)		#maximum rule
			predictions4=predictions4+w[i]*predictions						#weighted sum
	y_pred1=np.argmax(predictions1,axis=1)
	y_pred2=np.argmax(predictions2,axis=1)
	y_pred3=np.argmax(predictions3,axis=1)
	y_pred4=np.argmax(predictions4,axis=1)
	print('sum '+str(accuracy_score(dataset_y,y_pred1)))
	print('product '+str(accuracy_score(dataset_y,y_pred2)))	
	print('maximum '+str(accuracy_score(dataset_y,y_pred3)))
	print('weighted average'+str(accuracy_score(dataset_y,y_pred4)))

	# print(predictions1)
	# print('\n')
	# print(predictions2)
	# print('\n')
	# print(predictions3)
	# print('\n')
	# print(predictions4)
	# print('\n')

test('../Data/'+dataset)