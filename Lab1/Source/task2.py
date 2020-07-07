import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import seaborn as sns

#feeding the data Costomers.csv to data variable
data = pd.read_csv('Customers.csv')

# Null values condition check
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:5])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)
print(50*"==")

# Handling the null values if it has any
data = data.select_dtypes(include=[np.number]).interpolate().dropna()

# Using elbow method to find the good no. of clusters
wcss= []
#Taking only the last two columns that is spending and income
x = data.iloc[:,2:]
print(x)
#Visualising the data
sns.FacetGrid(x, height=4).map(plt.scatter, 'Annual Income (k$)', 'Spending Score (1-100)').add_legend()
plt.title('before clustering the data')
plt.show()

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.title('Elbow Graph')
plt.show()

# part - 2:
#From above plot we found that for no of clusters = 5 the graph is steadily decreasing
km =KMeans(n_clusters=5, random_state=0)
km.fit(x)
kmeans=km.predict(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print("Silhouette score is :",score)
x['res'] = y_cluster_kmeans
print(y_cluster_kmeans)
sns.FacetGrid(x,hue="res", height=4).map(plt.scatter, 'Annual Income (k$)', 'Spending Score (1-100)').add_legend()
plt.title('After clustering')
plt.show()


#Part -3
#From the first plot we can see that there the density of the values are
#formed at five different points so we can infer directly from the graph
#that we can use 5 clusters that is what we got from the elbow graph