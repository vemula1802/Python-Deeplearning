import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('./train_preprocessed.csv')
test_df = pd.read_csv('./test_preprocessed.csv')
X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
print(train_df['Survived'].value_counts(dropna='False'))

X_test = test_df.drop("PassengerId",axis=1).copy()
print(train_df[train_df.isnull().any(axis=1)])
#K Nearest Neighbour Alogirithm
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("KNN accuracy is:",acc_knn)
print(train_df[['Sex', 'Survived']].groupby(['Sex']).mean())

g = sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist,'Sex',bins=10)
plt.show()

