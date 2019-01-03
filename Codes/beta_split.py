import pandas as pd
import glob
import os
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif

def select_and_write(filename):
	print(filename)
	# Read dataset
	outfile='f_classif_'+filename
	nummFeatures=50


	df=pd.read_csv(filename,header=None)
	df = df.sample(frac=1).reset_index(drop=True)
	y=df.iloc[:,-1]
	df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
	

	from sklearn import preprocessing

	x = df.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	df = pd.DataFrame(x_scaled)

	X=df
	selector=SelectKBest(score_func=f_classif,k=100)
	X=selector.fit_transform(X,y)
	X=pd.DataFrame(X)
	X['class']=y
	X.to_csv('outfile1.csv',index=False)

	X=df
	selector=SelectKBest(score_func=mutual_info_classif,k=100)
	X=selector.fit_transform(X,y)
	X=pd.DataFrame(X)
	X['class']=y
	X.to_csv('outfile2.csv',index=False)

	X=df
	selector=SelectKBest(score_func=chi2,k=100)
	X=selector.fit_transform(X,y)
	X=pd.DataFrame(X)
	X['class']=y
	X.to_csv('outfile3.csv',index=False)

select_and_write('arrhythmia.data')
