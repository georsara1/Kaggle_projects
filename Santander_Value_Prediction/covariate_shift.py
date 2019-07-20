import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


print('Importing data...')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

## missing values
for i in train.columns:
    if train[i].dtype == 'object':
      train[i] = train[i].fillna(train[i].mode().iloc[0])
    if (train[i].dtype == 'int' or train[i].dtype == 'float'):
      train[i] = train[i].fillna(np.mean(train[i]))


for i in test.columns:
    if test[i].dtype == 'object':
      test[i] = test[i].fillna(test[i].mode().iloc[0])
    if (test[i].dtype == 'int' or test[i].dtype == 'float'):
      test[i] = test[i].fillna(np.mean(test[i]))

## label encoding
number = LabelEncoder()
for i in train.columns:
    if (train[i].dtype == 'object'):
      train[i] = number.fit_transform(train[i].astype('str'))
      train[i] = train[i].astype('object')

for i in test.columns:
    if (test[i].dtype == 'object'):
      test[i] = number.fit_transform(test[i].astype('str'))
      test[i] = test[i].astype('object')


## creating a new feature origin
train['origin'] = 0
test['origin'] = 1
training = train.drop('target',axis=1) #droping target variable

## taking sample from training and test data
training = training.sample(3200, random_state=12)
testing = test.sample(3000, random_state=11)

## combining random samples
combi = training.append(testing)
y = combi['origin']
combi.drop('origin',axis=1,inplace=True)

## modelling
model = RandomForestClassifier(n_estimators = 50, max_depth = 5,min_samples_leaf = 5)
drop_list = []
auc_score=[]
for i in combi.columns:
    score = cross_val_score(model,pd.DataFrame(combi[i]),y,cv=2,scoring='roc_auc')
    if (np.mean(score) > 0.8):
        drop_list.append(i)
    print(i,np.mean(score))
    auc_score.append(np.mean(score))


covar_scores = pd.DataFrame({'features': np.array(combi.columns), 'score': np.array(auc_score)})
covar_scores = covar_scores.sort_values(by = 'score', ascending= False)
to_exclude = covar_scores.features[:30]

del training
del testing
del train
del test
del combi