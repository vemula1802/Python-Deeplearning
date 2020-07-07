# Importing necessary libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

# Reading and loading the data set
# Data set considered here is cars.csv
data = pd.read_csv('cars.csv', delimiter=',', header=None, skiprows=1, names=['mpg','cylinders','cubicinches','hp','weightlbs','time-to-60','year','brand'])
print(data.head())
print('='*50)
# (a) Performing the data analysis on the data set

# Count the number of classes in the target 'brand' or top 3 correlated features are printed
print("The top 3 correlated features are given below")
print(data['brand'].value_counts(dropna=False))
print('='*50)

# Handling the Null Values
print("Displaying columns with null values")
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Features']
nulls.index.name = 'Nulls count'
print(nulls)
print("No null values are found")
# No nulls were found, so it is not necessary to delete any null values
print('='*50)

# Visualize data to analyze our feature correlations
import seaborn as sns
import matplotlib.pyplot as plt
sns.FacetGrid(data, hue='brand', height=4).map(plt.scatter, 'brand', 'mpg').add_legend()
plt.show()
sns.FacetGrid(data, hue='brand', height=4).map(plt.scatter, 'brand', 'cylinders').add_legend()
plt.show()
sns.FacetGrid(data, hue='brand', height=4).map(plt.scatter, 'brand', 'cubicinches').add_legend()
plt.show()
sns.FacetGrid(data, hue='brand', height=4).map(plt.scatter, 'brand', 'hp').add_legend()
plt.show()
sns.FacetGrid(data, hue='brand', height=4).map(plt.scatter, 'brand', 'weightlbs').add_legend()
plt.show()
sns.FacetGrid(data, hue='brand', height=4).map(plt.scatter, 'brand', 'time-to-60').add_legend()
plt.show()


# Encoding non-numeric features
from sklearn.preprocessing import LabelEncoder
data = data.apply(LabelEncoder().fit_transform)

# Split data into train and test
from sklearn.model_selection import train_test_split
x = data.drop(['brand'], axis=1)
y = data['brand']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=0)

# (b) Applying the three types of classifiers on the data set

# Naive Bayes method
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
nb = GaussianNB()
nb.fit(x_train, y_train)
# Evaluate model
#.score() for train data calculate the difference between y_train from model and accuracy measure y_train
score = nb.score(x_train, y_train)
print('Naive Bayes accuracy training score: ', score)
print('Classification report:')
y_pred = nb.predict(x_test)
print(classification_report(y_test, y_pred))
print()
print('='*50)

# KNN method
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# Setup arrays to store training and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the model
    knn.fit(x_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(x_train, y_train)

    # Compute accuracy on the test set
    test_accuracy[i] = knn.score(x_test, y_test)

#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
# At k=3, the value remains mostly unchanged
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
#Evaluate model
score = knn.score(x_train, y_train)
print('K-Neighbors accuracy training score: ', score)
print('Classification report:')
y_pred = knn.predict(x_test)
print(classification_report(y_test, y_pred))
print()
print('='*50)

#SVM method
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
#Evaluate model
score = svc.score(x_train, y_train)
print('Support Vector Machines score: ', score)
print('Classification report:')
y_pred = svc.predict(x_test)
print(classification_report(y_test, y_pred))
print('='*50)

print("From the three classifiers, knn gives the better result")
print('='*50)

# (c) Applying SVM with linear and non-linear kernel to check for the better performance

#SVM method (Linear)
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(x_train, y_train)
#Evaluate model
score = svc.score(x_train, y_train)
print('Support Vector Machines (Linear) score: ', score)
print('Classification report:')
y_pred = svc.predict(x_test)
print(classification_report(y_test, y_pred))
print('='*50)

#SVM method (Non-Linear)
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(x_train, y_train)
#Evaluate model
score = svc.score(x_train, y_train)
print('Support Vector Machines (Non-Linear) score: ', score)
print('Classification report:')
y_pred = svc.predict(x_test)
print(classification_report(y_test, y_pred))
print('='*50)

print("SVM with Linear Kernel gives the more accurate result when compared with non-linear kernel")
print('='*50)