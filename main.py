#importing libraries
from tkinter import YView

import numpy as np
import pandas as pd

#importing dataset
dataset=pd.read_csv("weatherAUS.csv")
X=dataset.iloc[:,[1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
Y=dataset.iloc[:,22].values
print(X)
print(Y)
#sklearn will only accept only two dimensional list
Y=Y.reshape(-1,1)


#Dealing with invalid dataset==there are some values like na-not a number to convert it we use this

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X=imputer.fit_transform(X)
Y=imputer.fit_transform(Y)
print(X)
print(Y)

#Encoding==machine learning alogorithm only deals with numeric data so use encoding to remove string data ex:city
#so train with other than numeric data we use encoding

from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()#for column 1 ie.,city
X[:,0]=le1.fit_transform(X[:,0])
le2=LabelEncoder()
X[:,4]=le2.fit_transform(X[:,4])
le3=LabelEncoder()
X[:,6]=le3.fit_transform(X[:,6])
le4=LabelEncoder()
X[:,7]=le4.fit_transform(X[:,7])
le5=LabelEncoder()
X[:,-1]=le5.fit_transform(X[:,-1])
le6=LabelEncoder()
Y=le6.fit_transform(Y)
print(X)
print(Y)

#Feature Scaling==to improve speed Of our training process
#scale is -3 to +3 generally

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
print(X)

#Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)#random state for containing no irrelvent data
print("\n\nTraining Data")
print(X_train)
print(Y_train)

#Training Model
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,random_state=0)
print(classifier.fit(X_train,Y_train))  # to perform training of training dataset(x_train,y_train)
print(classifier.score(X_train,Y_train))   #to test the accuracy of training data and labels

print(Y_test)
Y_test=Y_test.reshape(-1,1)
print(Y_test)
Y_pred=classifier.predict(X_test)
print(Y_pred)
Y_pred=le6.inverse_transform(Y_pred)
print(Y_pred)
print(Y_test)
Y_test=le6.inverse_transform(Y_test)
print(Y_test)
Y_test=Y_test.reshape(-1,1)
Y_pred=Y_pred.reshape(-1,1)
df=np.concatenate((Y_test,Y_pred),axis=1)#axis=1=>vertical,axis=0=>horizontal
dataframe=pd.DataFrame(df,columns=['Rain on Tommorrow','Prediction of rain'])
print(df)
print(dataframe)

#Calculating Accuracy
from sklearn.metrics import accuracy_score
# metrics is used to evaluate your machine learning algorithms
print(accuracy_score(Y_test,Y_pred))
#To improve accuracy_score increase n_estimators in RandomForestClassifier
dataframe.to_csv('prediction.csv')

