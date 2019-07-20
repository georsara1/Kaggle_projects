from nltk.tokenize import word_tokenize
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from nltk import FreqDist
import copy

#Import data and split into train and testdatasets
print('Importing data set...')
df_all = pd.read_csv("train.csv", sep=",",error_bad_lines=False)
df_all = df_all.drop(['id'], axis=1)
df_train, df_test = train_test_split(df_all, test_size = 0.2)

#Tokenize sentences
print("Tokenizing...")
#1. Train set
text_list_train = list(df_train['text'])
text_list_train_lower = [word.lower() for word in text_list_train]
tokenized_text_train = [word_tokenize(i) for i in text_list_train_lower]

#2. Test set
text_list_test = list(df_test['text'])
text_list_test_lower = [word.lower() for word in text_list_test]
tokenized_text_test = [word_tokenize(i) for i in text_list_test_lower]

#--------------------------Pre-processing train and test sets----------------------------------
print('Pre-processing Train set...')

vocab_size = 20000
dist_X = FreqDist(np.hstack(tokenized_text_train))
X_vocab = dist_X.most_common(vocab_size-1)

X_ix_to_word = [word[0] for word in X_vocab[:]] #Exclude top 200 most frequent words
X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}

tokenized_numbers_train = copy.deepcopy(tokenized_text_train)

for i, sentence in enumerate(tokenized_text_train):
    for j, word in enumerate(sentence):
        if word in X_word_to_ix:
            tokenized_numbers_train[i][j] = X_word_to_ix[word]
        else:
            tokenized_numbers_train[i][j] = 0


print('Pre-processing Test set...')
tokenized_numbers_test = copy.deepcopy(tokenized_text_test)

for i, sentence in enumerate(tokenized_text_test):
    for j, word in enumerate(sentence):
        if word in X_word_to_ix:
            tokenized_numbers_test[i][j] = X_word_to_ix[word]
        else:
            tokenized_numbers_test[i][j] = 0



print('Zero-padding in progress...')

#Bring both sets to same shape (Choose how many words to use)
max_words_in_sentence=30

# Zero-padding
tokenized_numbers_train = pad_sequences(tokenized_numbers_train, maxlen=max_words_in_sentence, padding='pre', dtype='int32')
tokenized_numbers_test = pad_sequences(tokenized_numbers_test, maxlen=max_words_in_sentence, padding='pre', dtype='int32')

#------------------------------End of Pre-processing----------------------------------------------------

#Define train and Test sets
train_x = np.array(tokenized_numbers_train)
train_x = np.delete(train_x, 29, 1)
train_y = np.array(df_train['author'])

test_x = np.array(tokenized_numbers_test)
test_x = np.delete(test_x, 29, 1)
test_y = np.array(df_test['author'])

encoder1 = LabelEncoder()
encoder1.fit(train_y)
encoded_train_Y = encoder1.transform(train_y)
dummy_train_y = np_utils.to_categorical(encoded_train_Y)
dummy_train_y.astype(int)

encoder2 = LabelEncoder()
encoder2.fit(test_y)
encoded_test_Y = encoder1.transform(test_y)
dummy_test_y = np_utils.to_categorical(encoded_test_Y)
dummy_test_y.astype(int)

l=vocab_size
inp=train_x.shape[1]

#Build an LSTM model
model = Sequential()
model.add(Embedding(l,96,input_length=inp))
model.add(LSTM(64, dropout=0.9, recurrent_dropout=0.2))
model.add(Dense(32, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
history = model.fit(train_x, dummy_train_y, validation_split=0.2 , epochs=8, batch_size=36, verbose=2)

# Final evaluation of the model
test_sentiments = model.predict(test_x)
#test_sentiments = pd.DataFrame(test_sentiments, columns=['EAP', 'HPL','MWS'])

test_sentiments[test_sentiments<0.5]=0
test_sentiments[test_sentiments>0.5]=1
test_sentiments.astype(int)

#Print accuracy
acc = accuracy_score(dummy_test_y,test_sentiments)
print('Overall accuracy:', acc)

#-----------------------------Plot metrics----------------------------

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