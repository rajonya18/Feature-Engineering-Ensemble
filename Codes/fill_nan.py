import pandas as pd
import numpy as np 

dataset_name='horse-colic'

df=pd.read_csv('../Data/'+dataset_name+'.data',header=None)
df.fillna(df.mean())
# df.replace('g',0,inplace=True)
# df.replace('b',1,inplace=True)
df.to_csv('../Data/'+dataset_name+'.csv',index=False)
