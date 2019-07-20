
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization
from keras import regularizers
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math

#Import data
df_train = pd.read_csv('train_v2.csv', low_memory=False)
df_test = pd.read_csv('test_v2.csv', low_memory=False)
df_submission = pd.read_csv('sampleSubmission.csv')

df_train.isnull().sum()

#Merge data sets
train_y = df_train.loss

df_train = df_train.drop(['loss'], axis = 1)

train_test_set = pd.concat([df_train, df_test], axis = 0)

#pre-processing
train_test_set = train_test_set.drop(['id'], axis = 1)

#Delete singular columns (columns with only a single value)
for col in train_test_set.columns:
    if len(train_test_set[col].unique()) == 1:
        train_test_set.drop(col, inplace = True, axis = 1)

##fill in null values
train_test_set = train_test_set.select_dtypes(['number'])
train_test_set = train_test_set.fillna(method = 'pad')
train_test_set = train_test_set.dropna()
train_test_set.isnull().sum().sum() #OK all imputed

train_test_set3 = (train_test_set-train_test_set.mean())/(train_test_set.max()-train_test_set.min())

#Split again in train and test sets
train_x = train_test_set3.iloc[:df_train.shape[0],:]
test_x = train_test_set3.iloc[df_train.shape[0]:,:]

import numpy as np
train_x = np.array(train_x)
test_x = np.array(test_x)
#train_x = train_x.sample(frac=1).reset_index(drop=True)

#train_x = preprocessing.scale(train_x)
#test_x = preprocessing.scale(test_x)

#Build model
model = Sequential()

model.add(Dense(256, input_dim=train_x.shape[1], kernel_initializer= 'normal', activation = 'tanh'))
model.add(BatchNormalization())
model.add(Dense(512, activation= 'tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'tanh'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop')

history = model.fit(train_x, train_y, validation_split = 0.2, epochs = 60, batch_size = 64)


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

predictions = model.predict(test_x)

predictions = [math.floor(p) for p in predictions]

plt.hist(predictions, bins = 40)
plt.show()

df_submission.loss = predictions

#df_submission.to_csv('NNsubmission.csv', index = False)
