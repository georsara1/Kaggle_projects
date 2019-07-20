
#Import necessary packages
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import ensemble

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
alldataOHE=pd.get_dummies(alldata)

#Split in train and test sets
trainX=alldataOHE.iloc[:4209,:-368] #Select all columns except from response variable
trainY=alldataOHE.iloc[:4209,368:369] #Select only response variable
testX=alldataOHE.iloc[4209:8419,:-368] #Select all columns except from response variable

#Fit model
clf = ensemble.GradientBoostingRegressor(n_estimators=200,max_depth=6)
clf.fit(trainX,trainY)
clf.score(trainX,trainY)

clf2=ensemble.AdaBoostRegressor()
clf2.fit(trainX,trainY)
print(clf2.score(trainX,trainY))

#Make prediction and write to file
predicted_sgd=clf.predict(testX)
predicted_ada=clf2.predict(testX)

pred=pd.DataFrame(predicted_sgd)
ids=pd.DataFrame(test['ID'])
final= pd.concat([ids, pred], axis=1, join='inner')
final.columns=['ID','y']
final.to_csv('sample_submission.csv',index=False,sep=',')

