import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.get_dataset_names()

data = sns.load_dataset("iris")
data

data.isnull().sum()

data.describe()

sns.pairplot(data)

x=data.iloc[:,:-1]
x

y = data["species"]
y

data.species=data["species"].map({"setosa":0,"versicolor":1,"virginica":2})
data

sns.heatmap(data.corr(),annot=True)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_scaled=ss.fit_transform(x)
x_scaled

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

y_predict=knn.predict(x_test)

from sklearn.metrics import accuracy_score,classification_report
accuracy = accuracy_score(y_predict,y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

print(classification_report(y_predict,y_test))
