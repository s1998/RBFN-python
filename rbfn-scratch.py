from sklearn.cluster import KMeans
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
from utils import *

# __init__ function will contain variables
# a separate function will be used for training 
# a separate function will be used for testing




class rbfn_model_tf:
	def __init__(self, indim, num_classes, no_of_clusters, x, beta):
		
		# define variables
		self.indim = indim
		self.num_classes = num_classes
		self.no_of_clusters = no_of_clusters
		self.weights = parameterWithAdamOpt(np.random.randn(no_of_clusters, num_classes))
		self.beta = beta
		self.cluster_centres = np.random.permutation(x)[:self.no_of_clusters, :].reshape(1, self.no_of_clusters, self.indim)
	def train(self, x, y):
		# print(self.weights.param)
		diff_2 = np.square(x.reshape(-1, 1, self.indim) / 128 - self.cluster_centres.reshape(1, -1, self.indim) / 128).sum(axis = 2)
		diff_2_exp = np.exp(diff_2)
		h = np.matmul(diff_2_exp, self.weights.param)
		h_exp = np.exp(h - np.max(h)) 
		print("69", h_exp)
		h_exp_rs = h_exp.sum(axis = 1) # row sum of h_exp
		print("70", h_exp_rs)
		for i in range(10):
			print("71", h_exp[0, i] / h_exp_rs[0])
		y_predicted = h_exp / h_exp_rs[:, None]

def accuracy(y_inp, y_pre):
	return sum(1.0*(np.argmax(y_inp, 1) == y_pre)) / (y_inp.shape[0])

def correct(y_inp, y_pre):
	return sum(1.0*(np.argmax(y_inp, 1) == y_pre))
	return sum(y_inp == y_pre)

model1 = rbfn_model_tf(784, 10, 100, mnist.train.next_batch(5000)[0], 1)
model2 = rbfn_model_tf(784, 10, 100, mnist.train.next_batch(5000)[0], 10)
model3 = rbfn_model_tf(784, 10, 100, mnist.train.next_batch(5000)[0], 100)
print("\n\nInitialized the model with random centres \n\n\n\n")

acc_now = 0
for batch_no in range(14000):
	batch = mnist.train.next_batch(500)
	# print(batch[0].shape)
	# print(batch[1].shape)
	if acc_now < 70:
		model1.train(batch[0], batch[1])
	# else:
	# 	model.train2(batch[0], batch[1])
