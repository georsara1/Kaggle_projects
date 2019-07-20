#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.utils.np_utils import to_categorical
from tqdm import tqdm

path = "full_dataset"

#Create dictionary
character_names = os.listdir(path)
idx = list(range(len(character_names)))
char_dict = dict(zip(idx,character_names))

## Import and pre-process images
image_set = []
label_set = []

for simpson in tqdm(os.listdir(path)):
    for img in os.listdir(path + '/' + simpson):
        character_name = simpson
        next_label = [label for label, character in char_dict.items() if character == character_name][0]
        next_img = cv2.imread(path + '/' + simpson + '/' + img)
        image_set.append(cv2.resize(next_img, (64, 64)))
        label_set.append(next_label)


#Split in Train and Test sets
X_train, X_test, Y_train, Y_test = train_test_split(image_set, label_set, test_size=0.2, random_state=22, shuffle=True)

#Convert list to array
X_train = np.array(X_train)
X_test = np.array(X_test)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-Hot Encoding of labels
Y_trainHot = to_categorical(Y_train, num_classes = 41)
Y_testHot = to_categorical(Y_test, num_classes = 41)


#----------------------Build a Convolutional Neural Network---------------------
#Select parameters
input_shape = (64, 64, 3)
batch_size = 32
num_classes = 41
epochs = 8

#Select architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(X_train, Y_trainHot, validation_split=0.2, epochs=epochs, batch_size=32, verbose=1)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


#------------------Predict on train and test sets-----------------
predictions_train = model.predict(X_train)
predictions_train_01 = np.argmax(predictions_train, axis=1)

predictions_test = model.predict(X_test)
predictions_test_01 = np.argmax(predictions_test, axis=1)


#------------------Print Confusion matrices------------------
#Train set
conf_mat_NN_train = confusion_matrix(Y_train, predictions_train_01)

plt.figure(figsize = (10,10))
plt.imshow(conf_mat_NN_train, cmap = plt.cm.coolwarm)
plt.title('Confusion Matrix of Train Set')
plt.colorbar()

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, conf_mat_NN_train[i, j],
                 horizontalalignment="center",
                 verticalalignment='center')

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, list(char_dict.values()), rotation=90)
plt.yticks(tick_marks, list(char_dict.values()))
plt.xlabel('Predicted character')
plt.ylabel('True character')
plt.tight_layout()
plt.show()


#Test set
conf_mat_NN_test = confusion_matrix(Y_test, predictions_test_01)

plt.figure(figsize = (10,10))
plt.imshow(conf_mat_NN_train, cmap = plt.cm.coolwarm)
plt.title('Confusion Matrix of Test Set')
plt.colorbar()

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, conf_mat_NN_test[i, j],
                 horizontalalignment="center",
                 verticalalignment='center')

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, list(char_dict.values()), rotation=90)
plt.yticks(tick_marks, list(char_dict.values()))
plt.xlabel('Predicted character')
plt.ylabel('True character')
plt.tight_layout()
plt.show()

#--------------------Print F1 score--------------------
f1_NN_train = f1_score(Y_train, predictions_train_01, average='micro')
print('F1 score on Train set:', round(f1_NN_train,3))

#Print F1 score
f1_NN_test = f1_score(Y_test, predictions_test_01, average='micro')
print('F1 score on Test set:', round(f1_NN_test,3))


