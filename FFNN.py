import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns # de gráficos

# Ignorar warnings
import warnings
warnings.filterwarnings('ignore')
df1 = pd.read_csv('creditcard.csv')
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df1.loc[:,df1.columns != 'Class'],
                                                    df1['Class'],
                                                    test_size=0.3)
from sklearn import preprocessing
preprocessParams = preprocessing.StandardScaler().fit(x_train)
x_train_normalized = preprocessParams.transform(x_train)
x_test_normalized = preprocessParams.transform(x_test)

 
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
NumerOfClasses = len(y_train.unique())
RN = Sequential() # create network structure
RN.add(Dense(10, input_shape = x_train_normalized.shape[1:], activation ='sigmoid'))
RN.add(Dense(NumerOfClasses, activation ='sigmoid'))
from keras.utils import to_categorical
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)
RN.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
trainedRN = RN.fit(x_train_normalized, to_categorical(y_train), epochs=2, verbose=1)
score = RN.evaluate(x_test_normalized, to_categorical(y_test),verbose=0)
#Predict
from sklearn.metrics import confusion_matrix
y_test_predicted = RN.predict(x_test_normalized)
y_test_predicted_index = np.argmax(y_test_predicted, axis=1)
y_test_index = y_test.values
def predict(a):
    if np.argmax(RN.predict(x_test[:85443]), axis=1)[a]==0:
        print('Not Fraud')
    else:
        print("fraud")
    