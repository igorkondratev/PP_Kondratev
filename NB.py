import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,auc,roc_auc_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
# Data Handling: Load CSV
df = pd.read_csv("creditcard.csv")
df["Time_Hr"] = df["Time"]/3600 # convert to hours
df = df.drop(['Time'],axis=1)
from sklearn.preprocessing import StandardScaler
df['scaled_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Amount'],axis=1)
def split_data(df, drop_list):
    df = df.drop(drop_list,axis=1)
    
    #test train split time
    from sklearn.model_selection import train_test_split
    y = df['Class'].values #target
    X = df.drop(['Class'],axis=1).values #features
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42, stratify=y)

  
    return X_train, X_test, y_train, y_test
def get_predictions(clf, X_train, y_train, X_test):
    # create classifier
    clf = clf
    # fit it to training data
    clf.fit(X_train,y_train)
    # predict using test data
    y_pred = clf.predict(X_test)
    
    
    #for fun: train-set predictions
    train_pred = clf.predict(X_train)
   
    return y_pred
def print_scores(y_test,y_pred):
    print(confusion_matrix(y_test,y_pred))
    
    print("accuracy : ", accuracy_score(y_test,y_pred))
from sklearn.naive_bayes import GaussianNB
# Case: do not drop anything
drop_list = []
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test,y_pred)