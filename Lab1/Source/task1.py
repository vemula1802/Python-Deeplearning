import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('creditcard.csv')
#Checking for Null values
print("before processing, null count is : " , sum(train_df.isnull().sum() != 0))
X_train = train_df.drop("Time",axis=1)
Y_train = train_df["Time"]

X_train, X_test, Y_train, Y_test= train_test_split(X_train, Y_train, test_size=0.3, random_state=0)


print(train_df[train_df.isnull().any(axis=1)])
#K Nearest Neighbour Alogirithm
knn = KNeighborsClassifier(n_neighbors = 3)
print("ihiin")
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("KNN accuracy is:",acc_knn)