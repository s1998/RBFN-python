from sklearn.cluster import KMeans
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

data = mnist.train.next_batch(60000)

points = data[0]

kmeans = KMeans(n_clusters=95, random_state=0).fit(points)
np.save("centroids_km.npy", kmeans.cluster_centers_)

