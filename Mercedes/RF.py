import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

rf=RandomForestRegressor(max_depth=3, n_estimators=10)
rf_enc=OneHotEncoder()
rf_lm=LogisticRegression()

train3=OneHotEncoder(train2)

for f in train.columns:
    if train[f].dtypes == 'object':
        cat_feat.append(f)
