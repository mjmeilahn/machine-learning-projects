
import pandas as pd

# THE PURPOSE IS TO FIND A DEPENDENT VARIABLE TO PREDICT (hence Clustering)

# Import the dataset
dataset = pd.read_csv('Mall_Customers.csv')

# Import values that can become the dependent variable
x = dataset.iloc[:, [3,4]].values
# print(x)

# We do not split data into Train & Test yet
# Because we are trying to find a dependent variable

# Use Elbow Method to find the correct amount of clusters
from sklearn.cluster import KMeans
wcss = []

# Run K-Means many times to find correct amount of clusters
for i in range(1, 11):
    # Use "init" to avoid the Random Initialization Trap
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Visually plot the Elbow Method
import matplotlib.pyplot as plt
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Choose the amount of clusters from the visual plot
# HARD CODE a value for "n_clusters"
# In this dataset "5" was the number we determined from Elbow Method

# Apply K-Means
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y = kmeans.fit_predict(x)
# print(y)

# Visualize the Five Clusters - HARD CODED
plt.scatter(x[y == 0,0], x[y == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y == 1,0], x[y == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y == 2,0], x[y == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(x[y == 3,0], x[y == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(x[y == 4,0], x[y == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='Centers')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
