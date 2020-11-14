#importing the libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
data= pd.read_csv("creditcard.csv")
No_of_frauds= len(data[data["Class"]==1])
No_of_normals = len(data[data["Class"]==0])

total= No_of_frauds + No_of_normals
Fraud_percent= (No_of_frauds / total)*100
Normal_percent= (No_of_normals / total)*100
fraud_index= np.array(data[data["Class"]==1].index)

#getting the list of normal indices from the full dataset
normal_index= data[data["Class"]==0].index

#choosing random normal indices equal to the number of fraudulent transactions
random_normal_indices= np.random.choice(normal_index, No_of_frauds, replace= False)
random_normal_indices= np.array(random_normal_indices)

# concatenate fraud index and normal index to create a list of indices
undersampled_indices= np.concatenate([fraud_index, random_normal_indices])

#use the undersampled indices to build the undersampled_data dataframe
undersampled_data= data.iloc[undersampled_indices, :]
#We have to scale the Amount feature before fitting our model to our dataset

sc= StandardScaler()
undersampled_data["scaled_Amount"]=  sc.fit_transform(undersampled_data.iloc[:,29].values.reshape(-1,1))

#dropping time and old amount column
#undersampled_data= undersampled_data.drop(["Time","Amount"])
X= undersampled_data.iloc[:, undersampled_data.columns != "Class"].values

y= undersampled_data.iloc[:, undersampled_data.columns == "Class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y)
#Using the gaussian kernel to build the initail model. Let us see if this is the best parameter later
classifier= SVC(C= 1, kernel= 'rbf', random_state= 0)
classifier.fit(X_train, y_train.ravel())
#Predict the class using X_test
y_pred = classifier.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)