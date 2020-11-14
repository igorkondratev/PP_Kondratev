import numpy as np 
import pandas as pd 
import keras

import matplotlib.pyplot as plt



df1 = pd.read_csv('creditcard.csv')
from sklearn.model_selection import train_test_split
# чтение файлаТаким образом вы делите свою выборку на тренировочную и тестовую часть. 
#Обучение будет происходит на тренировочной выборке, а на тестовой - проверка полученных "знаний". 
#test_size используется для разбиения выборки(в вашем случае будет 30% использовано на тест

x_train, x_test, y_train, y_test = train_test_split(df1.loc[:,df1.columns != 'Class'],#разделение датасета на тренировочный и тестовый поднаборы
                                                    df1['Class'],
                                                    test_size=0.3)
from sklearn import preprocessing
preprocessParams = preprocessing.StandardScaler().fit(x_train)#предобработка данных
x_train_normalized = preprocessParams.transform(x_train)
x_test_normalized = preprocessParams.transform(x_test)
def show_train_history(train_history,train,validation):#функция вывода графика обучения
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()
 
from keras import Sequential #модель нейросети
from keras.layers import Dense#функция бавления слоёв
from keras.optimizers import SGD#оптимизатор градиентного спуска
NumerOfClasses = len(y_train.unique())
RN = Sequential() # create network structure
RN.add(Dense(10, input_shape = x_train_normalized.shape[1:], activation ='relu'))#input_shape-входная форма данных
#RN.add(keras.layers.Dropout(0.9))
RN.add(Dense(10, activation='sigmoid'))
#RN.add(keras.layers.Dropout(0.9))
#RN.add(Dense(8, activation='relu'))#добавление слоя нейронной сети из 10 нейронов#RN.add(keras.layers.Dropout(0.1))
RN.add(Dense(NumerOfClasses, activation ='sigmoid'))
#RN.add(keras.layers.Dropout(0.9))
#функця активации сигмоид
from keras.utils import to_categorical#Преобразует вектор класса (целые числа) в двоичную классную матрицу
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)#градиентный спуск lr- скорость обучения,
RN.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])#компиляция нейронной сети
train_history = RN.fit(x_train_normalized, to_categorical(y_train), epochs=20, verbose=1)#тренировка нейронной сети
RN.summary()#вывод нейронной сети

#Predict
#from sklearn.metrics import confusion_matrix
y_test_predicted = RN.predict(x_test_normalized)#предиктинг нейронной сети
y_test_predicted_index = np.argmax(y_test_predicted, axis=1)
y_test_index = y_test.values

show_train_history(train_history,'accuracy','val_accuracy')#вывод графика

def predict(a):
    if np.argmax(y_test_predicted, axis=1)[a]==0:
        print('Not Fraud')
    else:
        print("fraud")