
#Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
sns.set_style("whitegrid")

#Import data
df = pd.read_csv('creditcard.csv', header = 0)


#---------------------------Pre-processing-------------------------
df.isnull().sum().sum() #No missing values

#Drop variables according to their importance (after the model has trained)
df = df.drop(['Time', 'Amount'
             ],
             axis = 1)


#Split in 75% train and 25% test set
train_df, test_df = train_test_split(df, test_size = 0.25, random_state= 1984)

#--------------------Create validation for early stopping of Light GBM train procedure------------
train_early_stop, valid_early_stop = train_test_split(train_df, test_size= 0.5, random_state= 7)

#-------------------------------------------------------------------------------------------------

train_y = train_df.Class
test_y = test_df.Class

train_x = train_df.drop(['Class'], axis = 1)
test_x = test_df.drop(['Class'], axis = 1)

#-------------------Build the Neural Network model-------------------
print('Building Neural Network model...')
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#adam = optimizers.adam(lr = 0.005, decay = 0.00000001)

model = Sequential()
model.add(Dense(24, input_dim=train_x.shape[1],
                kernel_initializer='uniform',
                #kernel_regularizer=regularizers.l2(0.02),
                activation="elu"))
#model.add(Dropout(0.5838))
# model.add(Dense(32,
#                 #kernel_regularizer=regularizers.l2(0.02),
#                 activation="tanh"))
# model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam', metrics = ['accuracy'])

history = model.fit(train_x, train_y, validation_split=0.2, epochs=15, batch_size=32
                   )

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Predict on test set
predictions_NN_prob = model.predict(test_x)
predictions_NN_prob = predictions_NN_prob[:,0]

predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

#Print accuracy
acc_NN = accuracy_score(test_y, predictions_NN_01)
print('Overall accuracy of Neural Network model:', acc_NN)

#Print Area Under Curve
false_positive_rate, recall, thresholds = roc_curve(test_y, predictions_NN_prob)
roc_auc = auc(false_positive_rate, recall)
plt.figure()
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()

#Print Confusion Matrix
cm = confusion_matrix(test_y, predictions_NN_01)
labels = ['No Default', 'Default']
plt.figure(figsize=(8,6))
sns.heatmap(cm,xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()
