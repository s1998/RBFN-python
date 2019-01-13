import numpy as np
points = np.load("points.npy")

import hdbscan

min_cluster_size = 4
max_points = 11000

hd = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, memory="./")
hd.fit(points[0:max_points])
print(hd)
print(min_cluster_size, max(hd.labels_))

labels = hd.labels_
centroids = []

empty = []
for i in range(100):
    if labels[labels == i].shape[0] == 0:
        continue
    centroids.append(points[:max_points][labels == i].sum(axis = 0) / labels[labels == i].shape[0])
    
print(len(centroids))

centroids = np.array(centroids)

print(centroids.shape)

np.save("hd_centroids.npy", centroids)

