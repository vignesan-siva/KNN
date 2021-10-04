
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("C:\\Users\\elcot\\Desktop\\machine learning tutorial\\classification\\wbcd.csv")

x=data.iloc[:,[2,4,5,9,22,24,25,29]].values
y=data.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder
labelen=LabelEncoder()
data['dia']=labelen.fit_transform(data['diagnosis'])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=19,metric='minkowski',p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)