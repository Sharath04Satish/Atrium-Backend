import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from math import sqrt

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical 

df = pd.read_csv("train5.csv", header=None)
X_train = df.values[:, :-1]
y_train = df.values[:, -1].astype(int)

df = pd.read_csv("test5.csv", header=None)
X_test = df.values[:, :-1]
y_test = df.values[:, -1].astype(int)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(1000, activation='relu', input_dim=187))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30)

pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   

pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    

model.save('atrium_model_11.h5')
