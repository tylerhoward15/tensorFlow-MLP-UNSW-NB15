import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import preprocessing
import pandas as pd


NB15_train = pd.read_csv('UNSW_NB15_training-set.csv', delimiter=',')
NB15_test = pd.read_csv('UNSW_NB15_testing-set.csv', delimiter=',')

training_nparray = NB15_train.to_numpy()  # This is necessary to correctly shape the array
testing_nparray = NB15_test.to_numpy()



# Preprocess
enc = preprocessing.OrdinalEncoder()

encoded_dataset = enc.fit_transform(training_nparray)  # All categorical features are now numerical
X_train = encoded_dataset[:, :-1]  # All rows, omit last column
y_train = np.ravel(encoded_dataset[:, -1:])  # All rows, only the last column

# Repeat preprocessing for test data
encoded_dataset = enc.fit_transform(testing_nparray)
X_test = encoded_dataset[:, :-1]
y_test = np.ravel(encoded_dataset[:, -1:])


input_dim = X_train.shape[1]
model = Sequential()
model.add(Dense(64, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=500,
          batch_size=128)
score = model.evaluate(X_test, y_test, batch_size=128)

total_datapoints = X_test.shape[0]
percent_correct = score[1] * 100
correct_datapoints = int(round(total_datapoints * percent_correct) / 100)
mislabeled_datapoints = total_datapoints - correct_datapoints


print("MultiLevelPerceptron Classifier results for UNSW-NB15 using TensorFlow and Keras:\n")
print("Total datapoints: %d\nCorrect datapoints: %d\nMislabeled datapoints: %d\nPercent correct: %.2f%%"
      % (total_datapoints, correct_datapoints, mislabeled_datapoints, percent_correct))