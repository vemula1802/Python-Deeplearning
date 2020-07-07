import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# reading the data
df = pd.read_csv('weather.csv')
# dropping precipitation type, formatted date, summary and daily summary columns
df = df.drop(['Precip Type', 'Formatted Date', 'Summary', 'Daily Summary'], axis=1)
# including only numerical features
numeric_features = df.select_dtypes(include=[np.number])
print(numeric_features)
# finding the null values in the data set
null_values = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
null_values.columns = ['Null Count']
null_values.index.name = 'Feature'
print(null_values)
# finding top 3 correlated features wrt temperature
correlation = numeric_features.corr()
print('The top 3 correlated features \n')
print(correlation['Temperature (C)'].sort_values(ascending=False)[:4], '\n')
# visualizing the data set columns
df[df.dtypes[(df.dtypes == "float64") | (df.dtypes == "int64")].index.values].hist(figsize=[11, 11])
plt.show()
# Plotting Temperature vs Apparent Temperature, Temperature vs Wind Speed, Temperature vs Wind Bearing
y_axis_data1 = df["Apparent Temperature (C)"]
y_axis_data2 = df["Wind Speed (km/h)"]
y_axis_data3 = df["Wind Bearing (degrees)"]
x_data = df["Temperature (C)"]
fig = plt.figure()
plt.rcParams['figure.figsize'] = (11, 11)
axis1 = fig.add_subplot(1, 3, 1)
axis2 = fig.add_subplot(1, 3, 2)
axis3 = fig.add_subplot(1, 3, 3)
axis1.plot(x_data, y_axis_data1, label='data 1')
axis2.plot(x_data, y_axis_data2, label='data 2')
axis3.plot(x_data, y_axis_data3, label='data 3')
axis1.set_xlabel('Temperature (C)')
axis1.set_ylabel('Apparent Temperature (C)')
axis2.set_xlabel('Temperature (C)')
axis2.set_ylabel('Wind Speed (km/h)')
axis3.set_xlabel('Temperature (C)')
axis3.set_ylabel('Wind Bearing (degrees)')
plt.show()

# Training Model with all features
x_train_data = df.drop("Temperature (C)", axis=1)
y_train_data = df["Temperature (C)"]
# splitting the data into train and test data
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x_train_data, y_train_data, test_size=0.3, random_state=0)
# training the model
lr = linear_model.LinearRegression()
model = lr.fit(x_train_data, y_train_data)
print("R Squared value is: ", model.score(x_test_data, y_test_data))
y_predictions = model.predict(x_test_data)
print('Root Mean Square Error is: ', mean_squared_error(y_test_data, y_predictions))

# Training Model with only top 3 correlated features
x_train_data = df[["Apparent Temperature (C)", "Visibility (km)", "Wind Bearing (degrees)"]]
y_train_data = df["Temperature (C)"]
# splitting the data into train and test data
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x_train_data, y_train_data, test_size=0.3, random_state=0)
# training the model with top 3 correlated features wrt temperature
lr = linear_model.LinearRegression()
model = lr.fit(x_train_data, y_train_data)
print("R Squared value is: ", model.score(x_test_data, y_test_data))
y_predictions = model.predict(x_test_data)
print('Root Mean Square Error is: ', mean_squared_error(y_test_data, y_predictions))