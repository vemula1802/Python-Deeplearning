import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
#Training the data
train = pd.read_csv('train.csv')
print(50*"==")
#Plot before removing the outlier data
plt.scatter(x=train['GarageArea'], y=train['SalePrice'])
plt.ylabel('Sale Price')
plt.xlabel('Garage area')
plt.show()
#Getting the mean value
garage_df = train.get('GarageArea')
print(garage_df)
z = np.abs(stats.zscore(garage_df))
print(z)
print(50*"==")
outliers = []
pos_to_drop = []
for i, score in enumerate(z):
    if score > 2.5:
        outliers.append((i, score))
        pos_to_drop.append(i)
print(outliers)

print(train)
train.drop(train.index[pos_to_drop], inplace=True)
print(train)
#Plot after removing the outlier data
plt.scatter(x=train['GarageArea'], y=train['SalePrice'])
plt.ylabel('Sale Price')
plt.xlabel('Garage area')
plt.show()
