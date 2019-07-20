#Code for the Rotten Tomatoes Kaggle contest

#Import libraries
print('Importing needed libraries...')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import Flatten, Dropout, Convolution1D
from keras.layers.embeddings import Embedding
from sklearn.metrics import roc_curve, auc

#Import data
print('Importing data...')
df_train = pd.read_csv('train.csv', sep = ',')
df_test = pd.read_csv('test.csv', sep = ',')

#Define train and Test sets
train_x = df_train.drop(["id","target","ps_calc_08","ps_calc_11", "ps_calc_12", "ps_calc_13", "ps_calc_14", "ps_car_03_cat"],axis=1)
train_y = df_train["target"]
test_x = df_test.drop(["id","ps_calc_08","ps_calc_11", "ps_calc_12", "ps_calc_13", "ps_calc_14", "ps_car_03_cat"], axis=1)

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)


#Build a Convolutional Neural Network model
print('Building the best model in the world...')
model = Sequential()
model.add(Embedding(2000, 256, input_length=51))
model.add(Convolution1D(128, 3, padding='same'))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

#------------------------------Fit the model-----------------------------
print('Fitting the best model in the world...')
history = model.fit(train_x, train_y, validation_split=0.2, epochs=3, batch_size=64, verbose=2)

#-----------------------------Plot metrics----------------------------
plt.figure(1)
# summarize history for accuracy
plt.subplot(121)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.35)
plt.show()

#----------------------------Predict test set and write to file---------------------------------
print('Predicting test set...')
test_sentiments = model.predict(test_x)
test_sentiments[test_sentiments<0.05]=0
test_sentiments[test_sentiments>0.05]=1
test_sentiments.astype(int)

id_df = pd.DataFrame(df_test['id'], dtype='int')
results_df = pd.DataFrame(test_sentiments, dtype= 'int', columns=['target'])

submission = pd.concat([id_df,results_df],axis=1)
submission.to_csv('submission.csv',index = False)

