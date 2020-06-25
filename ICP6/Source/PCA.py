import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#feeding the data cc.csv to data variable
data = pd.read_csv('CC.csv')
#Question 1
l=data.isnull().sum()
print(l)
#Replacing all the NaN values with their mean values
data = data.fillna(data.mean())
l=data.isnull().sum()
print(l)
print(50*"==")

# Using elbow method to find the good no. of clusters
wcss= []

x = data.iloc[:,1:17]
y = data.iloc[:,-1]

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,max_iter=100, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()

#Question 2:
#From above plot we found that for no of clusters = 3 the graph is steadily decreasing
km =KMeans(n_clusters=3, random_state=0)
km.fit(x)
kmeans=km.predict(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print("Silhouette score is :",score)

#QUESTION 3:
#Scaling

scaler = StandardScaler()
scaler.fit(x)
x_scaler= scaler.transform(x)
x_scaled = pd.DataFrame(x_scaler, columns =x.columns)
feature_scaling_score = metrics.silhouette_score(x_scaled, y_cluster_kmeans)
print("Silhouette score after scaling : ", feature_scaling_score)


#QUESTION 4
#Finding out PCA using two features
pca= PCA(2)
x_pca= pca.fit_transform(x_scaled)
print(50*"==")

#Bonus answer PCA + KMeans
km1 =KMeans(n_clusters=3, random_state=0)
km1.fit(x_pca)
y_cluster_kmeans1= km1.predict(x_pca)
pca_score = metrics.silhouette_score(x_pca, y_cluster_kmeans1)
print("PCA + Kmeans Score is :", pca_score)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = y_cluster_kmeans1)
plt.title('PCA + Kmeans')
plt.show()

#Bonus with PCA + Kmeans + scaling (For scaling i uses x_scalar instead of direct data(x))
x_pcascale = pca.fit_transform(x_scaler)
km = KMeans(n_clusters=3)
km.fit(x_pcascale)
Y_cluster_kmeans= km.predict(x_pcascale)
pca_means_scale_score = metrics.silhouette_score(x_pcascale, Y_cluster_kmeans)
print('PCA+KMEANS+ Scale score is:', pca_means_scale_score)
#PLotting the graph for bonus q1
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = Y_cluster_kmeans)
plt.title('PCA + Kmeans + Scaling')
plt.show()