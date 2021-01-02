import pandas as pd
import numpy as np
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

url="https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
s=requests.get(url).content
dataset=pd.read_csv(io.StringIO(s.decode('utf-8')))
#print(dataset)

#dataset = pd.read_csv("dataset.csv")
#print(dataset.head(10))

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Prediction for Test Data is:-\n",y_pred)
#print(confusion_matrix(y_test,y_pred))
print("Classification Report is:-\n",classification_report(y_test,y_pred))
print("Accuracy of Prediction is:-",accuracy_score(y_test, y_pred))
