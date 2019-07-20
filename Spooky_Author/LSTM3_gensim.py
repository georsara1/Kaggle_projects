from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from nltk import FreqDist
import copy

#Import data and split into train and testdatasets
print('Importing data set...')
df_all = pd.read_csv("train.csv", sep=",",error_bad_lines=False)
df_train, df_test = train_test_split(df_all, test_size = 0.2)

ids = df_test['id']
train_y = df_train['author']
test_y = df_test['author']

df_train = df_train.drop(['id'], axis=1)
df_test = df_test.drop(['id'], axis=1)

#Tokenize sentences

#1. Train set
text_list_train = list(df_train['text'])
#text_list_train_lower = [word.lower() for word in text_list_train]
#tokenized_text_train = [word_tokenize(i) for i in text_list_train]

#2. Test set
text_list_test = list(df_test['text'])
#text_list_test_lower = [word.lower() for word in text_list_test]
#tokenized_text_test = [word_tokenize(i) for i in text_list_test]

#--------------------------Pre-processing train and test sets----------------------------------
print('Pre-processing...')
#Parameter selection
max_words=13000
embed_dim = 300
max_words_in_sentence=25
word2vec = KeyedVectors.load_word2vec_format("C:/Users/georgios/Desktop/Python/word2vec/GoogleNews-vectors-negative300.bin", \
        binary=True)


print("Tokenizing...")
tokenizer = Tokenizer(num_words=max_words_in_sentence)
tokenizer.fit_on_texts(text_list_train + text_list_test)

train_sequences = tokenizer.texts_to_sequences(text_list_train)
test_sequences = tokenizer.texts_to_sequences(text_list_test)

word_index = tokenizer.word_index

# Bring both sets to same shape (zero-padding)
print('Zero-padding train and test sets...')
train_df = pad_sequences(train_sequences, maxlen=max_words_in_sentence, padding='pre')
test_df = pad_sequences(test_sequences, maxlen=max_words_in_sentence, padding='pre')

# prepare embedding matrix
print('Constructing embedding matrix...')
nb_words = min(max_words, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_dim))

j=0
for word, i in word_index.items():
    if word in word2vec and j<max_words:
        embedding_matrix[j,:] = word2vec[word]
        j+=1

#------------------------------End of Pre-processing----------------------------------------------------

#Define train and Test sets
train_x = np.array(train_df)
#train_x = np.delete(train_x, 29, 1)
train_y = np.array(df_train['author'])

test_x = np.array(test_df)
#test_x = np.delete(test_x, 29, 1)
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


inp=train_x.shape[1]

#Build an LSTM model
model = Sequential()
model.add(Embedding(nb_words,embed_dim, input_length=inp, weights=[embedding_matrix]))
model.add(LSTM(96, dropout=0.9, recurrent_dropout=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
history = model.fit(train_x, dummy_train_y, validation_split=0.2 , epochs=40, batch_size=48, verbose=2)

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