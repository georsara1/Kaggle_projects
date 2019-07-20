#Code for the Rotten Tomatoes Kaggle contest

#Import libraries
print('Importing needed libraries...')
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import itertools
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import copy

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import Flatten, Dropout, Convolution1D, MaxPooling1D, regularizers
from keras.layers.embeddings import Embedding
from keras.utils import np_utils

#Import data
df_train = pd.read_csv('train.tsv', sep = '\t')
df_test = pd.read_csv('test.tsv', sep = '\t')

PhraseID = df_test['PhraseId']
df_train = df_train.drop(df_train.columns[[0,1]], axis = 1)
df_test = df_test.drop(df_test.columns[[0,1]], axis = 1)

#Tokenize sentences
#1. Train set
text_list_train = list(df_train['Phrase'])
tokenized_text_train = [word_tokenize(i) for i in text_list_train]

#2. Test set
text_list_test = list(df_test['Phrase'])
tokenized_text_test = [word_tokenize(i) for i in text_list_test]

#Create vocabulary from both train and test sets
list_of_all_words_in_train_set = list(itertools.chain.from_iterable(tokenized_text_train))
list_of_all_words_in_test_set = list(itertools.chain.from_iterable(tokenized_text_test))
list_of_all_words = list_of_all_words_in_test_set+list_of_all_words_in_train_set
vocabulary =sorted(list(set(list_of_all_words)))

#Remove stopwords
stop_words = stopwords.words('english')
stop_words_updated = stop_words + (['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
vocabulary = [word for word in vocabulary if word not in stop_words_updated]


#-----------------------Pre-processing ---------------------------
print('Pre-processing Train set...')
tokenized_numbers_train = copy.deepcopy(tokenized_text_train)

i=-1
for list in tokenized_numbers_train:
    i=i+1
    j=-1
    for number in list:
        j = j + 1
        if tokenized_numbers_train [i][j] in vocabulary:
            tokenized_numbers_train[i][j] = vocabulary.index(number)
        else:
            tokenized_numbers_train[i][j] = 0

tokens_train = pd.DataFrame(tokenized_numbers_train, dtype='int32')
tokens_train = tokens_train.fillna(0)
tokens_train = tokens_train.astype(int)

print('Pre-processing Train set...')
tokenized_numbers_test = copy.deepcopy(tokenized_text_test)

i=-1
for list in tokenized_numbers_test:
    i=i+1
    j=-1
    for number in list:
        j = j + 1
        if tokenized_numbers_test[i][j] in vocabulary:
            tokenized_numbers_test[i][j]= vocabulary.index(number)
        else:
            tokenized_numbers_test[i][j] = 0

tokens_test = pd.DataFrame(tokenized_numbers_test, dtype='int32')
tokens_test = tokens_test.fillna(0)
tokens_test = tokens_test.astype(int)

#--------------------End of Pre-processing---------------------------

#Bring both sets to same shape (Choose how many words to use)
max_words_in_sentence=50

#Shorten or extend Train set to reach selected length
if tokens_train.shape[1]>max_words_in_sentence:
    tokens_train = tokens_train.drop(tokens_train.columns[[range(max_words_in_sentence,tokens_train.shape[1])]], axis=1)
else:
    for col in range(tokens_train.shape[1],max_words_in_sentence):
        tokens_train[col]=0

#Shorten or extend Test set to reach selected length
if tokens_test.shape[1] > max_words_in_sentence:
    tokens_test = tokens_test.drop(tokens_test.columns[[range(max_words_in_sentence, tokens_test.shape[1])]],
                                     axis=1)
else:
    for col in range(tokens_test.shape[1], max_words_in_sentence):
        tokens_test[col] = 0

#Define train and Test sets
train_x = np.array(tokens_train)
train_y = np.array(df_train['Sentiment'])

test_x = np.array(tokens_test)

#Transform target variable to One-Hot Encoding
encoder1 = LabelEncoder()
encoder1.fit(train_y)
encoded_train_Y = encoder1.transform(train_y)
dummy_train_y = np_utils.to_categorical(encoded_train_Y)
dummy_train_y.astype(int)
l=len(vocabulary)+1
inp=train_x.shape[1]

#Build a Convolutional Network model
model = Sequential()
model.add(Embedding(l, 128, input_length=inp))
#model.add(Convolution1D(96, 4, padding='same',activation='relu'))
#model.add(Convolution1D(96, 4, padding='same',activation='relu'))
#model.add(MaxPooling1D(pool_size=2,strides=None, padding='valid'))
#model.add(Dropout(0.1))
#model.add(Convolution1D(64, 3, padding='same',activation='relu'))
model.add(Convolution1D(128, 3, padding='valid',activation='relu'))
model.add(MaxPooling1D(pool_size=2,strides=None, padding='valid'))
model.add(Dropout(0.4))
model.add(Flatten())
#model.add(Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.02)))
#model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(0.02)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(train_x, dummy_train_y, validation_split=0.22, epochs=1, batch_size=64, verbose=2)

# Predict test set
test_sentiments = model.predict(test_x)
test_sentiments[test_sentiments<0.5]=0
test_sentiments[test_sentiments>0.5]=1
test_sentiments.astype(int)
Prediction = np.argmax(test_sentiments, axis=1)

#Build new dataframe and write to file to submit in contest
Submission_file = pd.DataFrame({'PhraseId': PhraseID, 'Sentiment': Prediction})
Submission_file.to_csv('Sample_Submission.csv', sep=',', index=False)


