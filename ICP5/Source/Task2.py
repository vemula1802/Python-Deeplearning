import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

train = pd.read_csv('winequality-red.csv')
#Checking for Null values
print("before processing, null count is : " , sum(train.isnull().sum() != 0))
# handling null or missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print("After removing, null  count is : " , sum(data.isnull().sum() != 0))

numeric_features = train.select_dtypes(include=[np.number])

corr = numeric_features.corr()
print(corr['quality'].sort_values(ascending=False)[1:4], '\n')

##Build a linear model
y = np.log(train.quality)
X = data.drop(['quality'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)


lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

##Evaluate the performance and visualize results
print ("R^2 is: \t", model.score(X_test, y_test))
predictions = model.predict(X_test)

print ('RMSE is: \t', mean_squared_error(y_test, predictions))

##visualizing the plot

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted quality')
plt.ylabel('Actual quality')
plt.title('Linear Regression Model')
plt.show()
