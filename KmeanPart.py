import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data for clustering
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Initialize K-Means with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit_predict(X)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200)
plt.title("K-Means Clustering")
plt.show()


###################################################################### PART 3