import glob
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.utils import to_categorical

#***************** Customisable parameters *******************************
# Split percentages
train_perc=0.4
val_perc=0.4
trval_perc=train_perc+val_perc
test_perc=0.2

num_classes=2

# number of features to select for each feature selection method
num_features=[50,50,50,0]
selection_fns=[chi2, f_classif, mutual_info_classif]
dataset_name='Hill_Valley'

# Number of epochs for each stage
num_epochs=[500,500,500,750]
# Batch size for each stage
batch_size=[15,15,15,10]
# Layers for each MLP
layer_det=[[100,70,35,10],
			[100,70,35,10],
			[100,70,35,10],
			[15,10,10,6]]
#*************************************************************************



def main(dataset_name):
	# The main function
	
	splitting_and_feature_selection('../Data/'+dataset_name)

	training_module('../Data/'+dataset_name)

	prediction_module('../Data/''../Data/'+dataset_name)

	second_training('../Data/'+dataset_name)

	#************** TESTING PHASE **********************************
	test('../Data/'+dataset_name)
	#***************************************************************

	print('Done.....')

def feature_select(dataset_name, dataset_X, dataset_y, filter_method, num_of_features, id):
	# Function which takes input the dataset and filter method and selects the number of features
	# It writes these set of features to a csv file
	feature_names = list(dataset_X.columns.values)
	selector=SelectKBest(score_func=filter_method,k=num_of_features)
	X=selector.fit_transform(dataset_X,dataset_y)

	# Find out the features which are selected
	mask = selector.get_support() #list of booleans
	new_features = [] # The list of your K best features

	for bool, feature in zip(mask, feature_names):
		if bool:
			new_features.append(feature)

	X=pd.DataFrame(X,columns=new_features)
	X['class']=dataset_y
	
	X.to_csv(dataset_name+'_trval'+str(id)+'.csv',index=False)
	# Also return the features which are selected so that the test set can also be prepared
	return new_features

def split(dataset_X, dataset_y, split_perc1, split_perc2):
	# Split the dataset
	# Returns train+validation and test sets
	train_X, test_X, train_y, test_y = train_test_split(dataset_X, dataset_y, train_size=split_perc1, test_size=split_perc2)
	return train_X, train_y, test_X, test_y

def train(train_X, train_y, test_X, test_y, num_of_features, classes, layer_det, id, num_epochs=30, batch_size=100):
	# Train a neural network
	# Saves a model to disk

	train_y = to_categorical(train_y)  #converts to one hot
	test_y = to_categorical(test_y)

	model = Sequential()
	model.add(Dense(layer_det[0], input_dim=num_of_features, activation='relu'))
	model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
	model.add(Dense(layer_det[1], activation='relu'))
	model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
	model.add(Dense(layer_det[2], activation='relu'))
	model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
	model.add(Dense(layer_det[3], activation='relu'))
	model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
	model.add(Dense(classes, activation='softmax'))
		
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


	# Fit the model
	model.fit(train_X, train_y, epochs=num_epochs, batch_size=batch_size)
	score = model.evaluate(train_X, train_y, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
	model.save('model'+str(id)+'.h5')  # creates a HDF5 file 'my_model.h5'

	score = model.evaluate(test_X,test_y, verbose=0)
	print("model"+str(id)+"====== %s: %.2f%%" % (model.metrics_names[1], score[1]*100))
	

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
			predictions=predict(dataset_X,dataset_y, (i+1))
		else:
			predictions=np.concatenate((predictions, predict(dataset_X,dataset_y, (i+1))),axis=1)


	column_headers=[i for i in range(num_classes*3)]
	test2 = pd.DataFrame(predictions, columns=column_headers)
	test2['class']=dataset_y
	# Write the predictions to csv file
	test2.to_csv(dataset_name+'_test4.csv',index=False)

	test2_y=test2['class']
	test2_y = to_categorical(test2_y)  #converts to one hot
	test2_X=test2.drop('class',axis=1)

	print('Testing model of second stage.....')
	# Load the model
	model_ensemble = load_model('model4.h5')
	score = model_ensemble.evaluate(test2_X, test2_y, verbose=0)
	print("model_ensemble====== %s: %.2f%%" % (model_ensemble.metrics_names[1], score[1]*100))


def splitting_and_feature_selection(dataset_name):
	# Load the dataset
	#************* SPLITTING AND FEATURE SELECTION *********************************** 
	print('Reading dataset....'+dataset_name)
	dataset=pd.read_csv(dataset_name+'.csv')

	print(np.unique(dataset['class'].values))
	dataset['class']=pd.Categorical(dataset['class'])
	dataset['class']=dataset['class'].cat.codes


	# Separate into X and y
	dataset_y=dataset['class']
	dataset_X=dataset.drop('class',axis=1)
	print(np.unique(dataset_y.values))
	print('Normalising dataset.....')
	x = dataset_X.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	dataset_X = pd.DataFrame(x_scaled, columns=dataset_X.columns)

	print('Splitting dataset....')
	# Split dataset into train+validation and test set
	trval_X, trval_y, test_X, test_y = split(dataset_X,dataset_y, trval_perc, test_perc)

	# Now select the features
	for i in range(3):
		# Select features from train+val
		print('Selecting '+str(num_features[i])+' features using '+str(selection_fns[i]))
		trval_y.reset_index(drop=True, inplace=True)
		new_features=feature_select(dataset_name, trval_X, trval_y, selection_fns[i] ,num_features[i], (i+1))
		# Select the same features from test set
		# features_to_be_deleted=[cols in test_X.columns not in new_features]

		print('New features are '+str(new_features))

		print('Select same feature sets from test data')
		tmp_test_X=test_X[new_features]
		tmp_test_X['class']=test_y

		print('Writing test file......')
		#Selected test features written to file
		tmp_test_X.to_csv(dataset_name+'_test'+str(i+1)+'.csv',index=False)

	# Split into train and validation set
	# For this we combine the trval for all 3 filter method into one dataframe and split it
	trval1=pd.read_csv(dataset_name+'_trval1.csv')
	trval2=pd.read_csv(dataset_name+'_trval2.csv')
	trval3=pd.read_csv(dataset_name+'_trval3.csv')


	print('Concatenating dataframes.....')
	# Now concatenate the dataframes
	trval1.drop('class', axis=1, inplace=True)
	trval2.drop('class', axis=1, inplace=True)

	trval_concat=pd.concat([trval1,trval2,trval3],axis=1, ignore_index=False)

	trval_concat_y=trval_concat['class']
	trval_concat.drop('class',axis=1,inplace=True)

	print('Split into train and validation.....')
	# Now split the dataframe
	train_X,train_y,val_X, val_y =split(trval_concat,trval_concat_y,0.5,0.5)

	cumulative_features=[0]*4;
	
	for i in range(1,4):
		cumulative_features[i]=cumulative_features[i-1]+num_features[i-1]

	#*********** WRITE TRAIN **********************
	# Now split the dataframes by columns
	for i in range(3):
		X=train_X.iloc[:,cumulative_features[i]:cumulative_features[i+1]]
		X['class']=train_y

		print('Writing train file '+str(i+1)+'....')
		# Write train set to file
		X.to_csv(dataset_name+'_train'+str(i+1)+'.csv',index=False)
	#***********************************************

	#*********** WRITE VALIDATION **********************
	for i in range(3):
		X=val_X.iloc[:,cumulative_features[i]:cumulative_features[i+1]]
		X['class']=val_y

		print('Writing validation file '+str(i+1)+'....')
		# Write validation set to file
		X.to_csv(dataset_name+'_val'+str(i+1)+'.csv',index=False)
	#***************************************************	
	#***********************************************

def training_module(dataset_name):
	#************** TRAINING *************************************
	# Now train the model
	print('Start training now...')
	for i in range(3):
		train_X=pd.read_csv(dataset_name+'_train'+str(i+1)+'.csv')
		train_y=train_X['class']
		train_X.drop('class',axis=1,inplace=True)

		test_X = pd.read_csv(dataset_name+'_test'+str(i+1)+'.csv')
		test_y = test_X['class']
		test_X.drop('class',axis=1,inplace=True)

		print('Training model '+str(i+1))
		train(train_X,train_y,test_X,test_y,num_of_features=num_features[i],classes=num_classes, layer_det=layer_det[i], id=(i+1), num_epochs=num_epochs[i], batch_size=batch_size[i])
	#***************************************************

def prediction_module(dataset_name):
	#************** PREDICTION *************************************
	for i in range(3):
		dataset_X = pd.read_csv(dataset_name+'_val'+str(i+1)+'.csv')
		dataset_y = dataset_X['class']
		dataset_X.drop('class', axis=1, inplace=True)
		# Predict and concatenate
		if (i==0):
			predictions=predict(dataset_X,dataset_y, (i+1))
		else:
			predictions=np.concatenate((predictions,predict(dataset_X,dataset_y, (i+1))),axis=1)


	column_headers=['a'+str(i) for i in range(num_classes*3)]
	train2 = pd.DataFrame(predictions, columns=column_headers)
	train2['class']=dataset_y

	print('Writing predictions to csv file.....')
	# Write the predictions to csv file
	train2.to_csv(dataset_name+'_train2.csv',index=False)
	#***************************************************************

def second_training(dataset_name):
	#************* SECOND TRAINING PHASE ***************************
	train2=pd.read_csv(dataset_name+'_train2.csv')
	train2_y=train2['class']
	train2_X=train2.drop('class',axis=1)

	print('Training model of second stage....')
	train(train2_X,train2_y,train2_X,train2_y,num_of_features=num_classes*3,classes=num_classes, layer_det=layer_det[3], id=4, num_epochs=num_epochs[3], batch_size=batch_size[3])
	#***************************************************************


main(dataset_name)