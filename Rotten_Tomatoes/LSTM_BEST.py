#Code for the Rotten Tomatoes Kaggle contest

#Import libraries
print('Importing needed libraries...')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Import data
df_train = pd.read_csv('train.tsv', sep = '\t')
df_test = pd.read_csv('test.tsv', sep = '\t')

PhraseID = df_test['PhraseId']#[:1000] #keep to build the submission file
df_train = df_train.drop(df_train.columns[[0,1]], axis = 1)
df_test = df_test.drop(df_test.columns[[0,1]], axis = 1)

#df_train = df_train.iloc[:4200,]
#df_test = df_test.iloc[:1000]

max_words=10000
text_list=list(df_train['Phrase'])+list(df_test['Phrase'])
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text_list)
sequences = tokenizer.texts_to_sequences(text_list)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

max_length = len(max(sequences,key=len))
data = pad_sequences(sequences, maxlen=30)

#Define train and Test sets
train_x = data[:156060,]
train_y = np.array(df_train['Sentiment'][:156060])

test_x = data[156060:,]

#Transform target variable to One-Hot Encoding
encoder1 = LabelEncoder()
encoder1.fit(train_y)
encoded_train_Y = encoder1.transform(train_y)
dummy_train_y = np_utils.to_categorical(encoded_train_Y)
dummy_train_y.astype(int)

#Build an LSTM model
l=len(word_index)+1
inp=train_x.shape[1]

model = Sequential()
model.add(Embedding(l, 64,input_length=inp))
model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.1))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
print('Fitting the best model in the world...')
model.fit(train_x, dummy_train_y,validation_split=0.22, epochs=2, batch_size=32, verbose=1)

# Predict test set
test_sentiments = model.predict(test_x)
test_sentiments[test_sentiments<0.5]=0
test_sentiments[test_sentiments>0.5]=1
test_sentiments.astype(int)
Prediction = np.argmax(test_sentiments, axis=1)

#Build new dataframe and write to file to submit in contest
Submission_file = pd.DataFrame({'PhraseId': PhraseID, 'Sentiment': Prediction})
Submission_file.to_csv('Sample_Submission.csv', sep=',', index=False)


