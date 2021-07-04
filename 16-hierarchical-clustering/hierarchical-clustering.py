
import pandas as pd

# THE PURPOSE IS TO FIND A DEPENDENT VARIABLE TO PREDICT (hence Clustering)

# Import the dataset
dataset = pd.read_csv('Mall_Customers.csv')

# Import values that can become the dependent variable
x = dataset.iloc[:, [3,4]].values
# print(x)

# We do not split data into Train & Test yet
# Because we are trying to find a potential dependent variable

# Use a Dendogram to find the optimal number of clusters
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
dendrogram = hierarchy.dendrogram(hierarchy.linkage(x, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Choose the amount of clusters from the Dendogram
# HARD CODE a value for "n_clusters"
# In this dataset "5" was the number we determined

# Apply Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y = hc.fit_predict(x)
# print(y)

# Visualize the Five Clusters - HARD CODED
plt.scatter(x[y == 0,0], x[y == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y == 1,0], x[y == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y == 2,0], x[y == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(x[y == 3,0], x[y == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(x[y == 4,0], x[y == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
