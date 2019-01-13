from rbfn import *

cluster_centres_count = 100
cluster_centres = np.random.permutation(mnist.train.next_batch(5000)[0])[:cluster_centres_count, :]

# initializing rbfn model
model1 = rbfn_model_tf(784, 10, cluster_centres_count, cluster_centres)
model2 = rbfn_model_tf(784, 10, cluster_centres_count, cluster_centres, True)
print("\n\nInitialized the model with random centres \n\n\n\n")

output = {}
accs, loss = model_trainer(model1, "fixed cluster centres")
output["fixed cluster centres"] = (accs, loss)
print("\n\nVariable_cc is true now \n")
accs, loss = model_trainer(model2,  "variable cluster centres")
output["variable cluster centres"] = (accs, loss)

import pickle
with open(os.path.join("..", "output_data", "compare_variable_centres.pkl"), "wb") as f:
  pickle.dump(output, f)
