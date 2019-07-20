
#Import necessary packages
import pandas as pd
import numpy as np
from sklearn import tree

#Import train and test datasets
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
test['y']=1.00

#Concatenate dataframes to perform EDA and cleaning
cols = train.columns.tolist()
cols = cols[0:1] + cols[2:379] + cols[1:2]
train = train[cols]
alldata=pd.concat([train,test],ignore_index=True)

#OneHotEncoding
alldata.drop(['ID'],inplace=True,axis=1) #Remove "ID" variable
alldataOHE=pd.get_dummies(alldata.iloc[:,:])

#Split in train and test sets
trainX=alldataOHE.iloc[:4209,:-1]
trainY=alldataOHE['y'][:4209]

testX=alldataOHE.iloc[4209:8419,:-1]

#Fit model
model=tree.DecisionTreeRegressor(max_depth=4,max_features=20,max_leaf_nodes=4)
model.fit(trainX,trainY)
model.score(trainX,trainY)

#Make prediction and write to file
predicted=model.predict(testX)
pred=pd.DataFrame(predicted)
ids=pd.DataFrame(list(range(1,len(pred)+1)))
final= pd.concat([ids, pred], axis=1, join='inner')
final.columns=['ID','y']
final.to_csv("sample_submission.csv",index=False)


