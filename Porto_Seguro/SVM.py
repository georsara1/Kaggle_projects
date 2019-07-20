#Code for the Rotten Tomatoes Kaggle contest

#Import libraries
print('Importing needed libraries...')
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import itertools
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold

#Import data
print('Importing data...')
df_train = pd.read_csv('train.csv', sep = ',')
df_test = pd.read_csv('test.csv', sep = ',')

#Check for NaNs
nans_df_train=pd.isnull(df_train).sum()
nans_df_test=pd.isnull(df_test).sum()

df_train.dtypes.value_counts()

df_train2 = pd.get_dummies(df_train)



pca = PCA(n_components=36,whiten=True)
pca = pca.fit(df_train2)

#Define train and Test sets
train_x = df_train.drop(["id","target"],axis=1)
train_x = np.array(train_x)
train_y = df_train["target"]
train_y = np.array(train_y)

test_x = df_train.drop("id", axis=1)
test_x = np.array(test_x)

#Build model
print("Training model...")
model = svm.SVC(kernel='linear', C=1, gamma=1)
model.fit(train_x,train_y)
model.score(train_x,train_y)

print("Making predictions..")
predicted= model.predict(test_x)