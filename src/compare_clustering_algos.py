from rbfn import *
import os 

centroids_km = np.load(os.path.join('..', 'MNIST_data', "centroids_km.npy")).astype(np.float32)
centroids_hd = np.load(os.path.join('..', 'MNIST_data', "centroids_hd.npy")).astype(np.float32)
centroids_sc = np.load(os.path.join('..', 'MNIST_data', "centroids_sc.npy")).astype(np.float32)

print("Centroids k means shape : ", centroids_km.shape)
print("Centroids spectral clustering shape : ", centroids_sc.shape)
print("Centroids HDBSCAN shape : ", centroids_hd.shape)

cluster_centres_count = min(centroids_km.shape[0], centroids_hd.shape[0], centroids_sc.shape[0])
print(cluster_centres_count)

output = {}

# initializing rbfn model
model1 = rbfn_model_tf(784, 10, cluster_centres_count, centroids_km[:cluster_centres_count, :], True)
print("\n\nInitialized the model with k means centres \n\n\n\n")
accs, loss = model_trainer(model1, "k means cluster centres")
output["k means cluster centres"] = (accs, loss)
tf.reset_default_graph()

model2 = rbfn_model_tf(784, 10, cluster_centres_count, centroids_sc[:cluster_centres_count, :], True)
print("\n\nInitialized the model with spectral clustering centres \n\n\n\n")
accs, loss = model_trainer(model2, "spectral clustering cluster centres")
output["spectral clustering cluster centres"] = (accs, loss)
tf.reset_default_graph()

model3 = rbfn_model_tf(784, 10, cluster_centres_count, centroids_hd[:cluster_centres_count, :], True)
print("\n\nInitialized the model with HDBSCAN centres \n\n\n\n")
accs, loss = model_trainer(model3, "HDBSCAN cluster centres")
output["HDBSCAN cluster centres"] = (accs, loss)
tf.reset_default_graph()

import pickle
with open(os.path.join("..", "output_data", "compare_cluster_algos_2.pkl"), "wb") as f:
  pickle.dump(output, f)
