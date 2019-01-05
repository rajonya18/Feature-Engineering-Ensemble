import pandas as pd
import numpy as np 

dataset_name='horse-colic'

df1=pd.read_csv(dataset_name+'_Train.data',header=None)
df2=pd.read_csv(dataset_name+'_Test.data',header=None)

df=df1.append(df2,ignore_index=True)
df.to_csv('../Data/'+dataset_name+'.data',index=False,sep=',')
