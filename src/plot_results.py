import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns

sns.set()

with open(os.path.join("..", "output_data", "compare_cluster_algos.pkl"), "rb") as f:
  output = pickle.load(f)

for k in output:
  accs = output[k][0]
  sns.lineplot(
    [i*10 for i in range(len(accs))][::10], accs[::10], 
    label = k + ", max : " + str(max(accs)))

plt.xlabel("minibatch count")
plt.ylabel("accuracy")
plt.title("Accuracy for different clustering algorithms")
# plt.show()
plt.savefig(os.path.join("..", "images", "Accuracy_for_different_clustering_algorithms.png"))
plt.clf()

for k in output:
  loss = output[k][1]
  sns.lineplot(
    [i*10 for i in range(len(loss))][::100], loss[::100], 
    label = k + ", min : " + str(min(loss)))

plt.xlabel("minibatch count")
plt.ylabel("loss")
plt.title("Loss for different clustering algorithms")
plt.savefig(os.path.join("..", "images", "Loss_for_different_clustering_algorithms.png"))
plt.clf()
# plt.show()

with open(os.path.join("..", "output_data", "compare_cluster_algos_2.pkl"), "rb") as f:
  output = pickle.load(f)

for k in output:
  accs = output[k][0]
  sns.lineplot(
    [i*10 for i in range(len(accs))][::10], accs[::10], 
    label = k + ", max : " + str(max(accs)))

plt.xlabel("minibatch count")
plt.ylabel("accuracy")
plt.title("Accuracy for different clustering algorithms with variable centres")
# plt.show()
plt.savefig(os.path.join("..", "images", "Accuracy_for_different_clustering_algorithms_2.png"))
plt.clf()

for k in output:
  loss = output[k][1]
  sns.lineplot(
    [i*10 for i in range(len(loss))][::100], loss[::100], 
    label = k + ", min : " + str(min(loss)))

plt.xlabel("minibatch count")
plt.ylabel("loss")
plt.title("Loss for different clustering algorithms")
plt.savefig(os.path.join("..", "images", "Loss_for_different_clustering_algorithms_2.png"))
plt.clf()


with open(os.path.join("..", "output_data", "compare_variable_centres.pkl"), "rb") as f:
  output = pickle.load(f)

for k in output:
  accs = output[k][0]
  sns.lineplot(
    [i*10 for i in range(len(accs))][::10], accs[::10], 
    label = k + ", max : " + str(max(accs)))

plt.xlabel("minibatch count")
plt.ylabel("accuracy")
plt.title("Accuracy for fixed vs changing centres")
# plt.show()
plt.savefig(os.path.join("..", "images", "Accuracy_for_variable_cluster_centres.png"))
plt.clf()

for k in output:
  loss = output[k][1]
  sns.lineplot(
    [i*10 for i in range(len(loss))][::100], loss[::100], 
    label = k + ", min : " + str(min(loss)))

plt.xlabel("minibatch count")
plt.ylabel("loss")
plt.title("Loss for fixed vs changing centres")
plt.savefig(os.path.join("..", "images", "Loss_for_variable_cluster_centres.png"))
# plt.show()
