import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from PIL import Image

image1 = Image.open('image1.png')
image2 = Image.open('image2.png')

image1_array = np.array(image1)
image2_array = np.array(image2)

pixels1 = image1_array.reshape(-1, 3)
pixels2 = image2_array.reshape(-1, 3)

sse = []
k_values = range(1, 20)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(pixels1)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(k_values, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method for Optimal k')
#plt.show()
plt.close()

optimal_k = 15

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
kmeans.fit(pixels1)
centroids = kmeans.cluster_centers_
labels1 = kmeans.labels_

compressed_image1 = centroids[labels1].reshape(image1_array.shape).astype(np.uint8)

knn = NearestNeighbors(n_neighbors=1)
knn.fit(centroids)
nearest_centroids_indices = knn.kneighbors(pixels2, return_distance=False)
compressed_image2 = centroids[nearest_centroids_indices.flatten()].reshape(image2_array.shape).astype(np.uint8)

Image.fromarray(compressed_image1).save('image1_done.png')
Image.fromarray(compressed_image2).save('image2_done.png')
