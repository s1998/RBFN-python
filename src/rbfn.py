from sklearn.cluster import KMeans
import numpy as np 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
mnist = input_data.read_data_sets(os.path.join('..', 'MNIST_data'), one_hot=True)

# __init__ function will make the computation graph
# a separate function will be used for training 
# a separate function will be used for testing




class rbfn_model_tf:
	def __init__(self, indim, num_classes, no_of_clusters, x, variabl_cc = False):
		
		# define variables
		self.indim = indim
		self.num_classes = num_classes
		self.no_of_clusters = no_of_clusters
		self.weights = tf.Variable(tf.random_uniform((no_of_clusters, num_classes)))
		self.beta = tf.Variable(-1.0)
		
		# get clusters
		if variabl_cc:
			self.cluster_centres = tf.Variable(x.reshape(1, self.no_of_clusters, self.indim))
		else:
			self.cluster_centres = tf.constant(x.reshape(1, self.no_of_clusters, self.indim))

		# define placeholders and define computation graph
		self.x_input = tf.placeholder(tf.float32, [None, indim])
		self.x_input_r = tf.reshape(self.x_input, (-1, 1, 784))
		self.x_input_d = self.x_input_r - self.cluster_centres
		self.x_input_d_2 = tf.square(self.x_input_d)
		self.x_input_s = tf.reduce_sum(self.x_input_d_2, axis = 2) 

		self.x_input_g = tf.exp(self.beta * self.x_input_s)
		self.y_input = tf.placeholder(tf.uint8, [None, 10])
		self.y_input_o = self.y_input
		self.y_predicted = tf.matmul(self.x_input_g, self.weights)
		
		self.loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.y_predicted, labels = self.y_input_o)
		self.loss_t = tf.reduce_mean(self.loss)

		self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
		self.trainer = self.optimizer.minimize(self.loss_t)
		
		self.optimizer2 = tf.train.AdamOptimizer(learning_rate = 0.001)
		self.trainer2 = self.optimizer2.minimize(self.loss_t)

		self.sess = tf.Session()
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)

	def train(self, x, y):
		result = self.sess.run([self.trainer, self.loss_t], feed_dict={self.x_input: x, self.y_input:y})
		return result

	def train2(self, x, y):
		result = self.sess.run([self.trainer2, self.loss_t], feed_dict={self.x_input: x, self.y_input:y})
		return result

	def predict(self, x):
		result = self.sess.run(self.y_predicted, feed_dict={self.x_input: x})
		result = np.argmax(result, axis = 1)
		result = np.reshape(result, [-1])
		return result

def accuracy(y_inp, y_pre):
	return sum(1.0*(np.argmax(y_inp, 1) == y_pre)) / (y_inp.shape[0])

def correct(y_inp, y_pre):
	return sum(1.0*(np.argmax(y_inp, 1) == y_pre))
	return sum(y_inp == y_pre)

def model_trainer(model, name):
	all_accs = []
	all_loss = []
	acc_now = 0	
	for batch_no in range(10000):
		batch = mnist.train.next_batch(128)
		if acc_now < 70:
			_, l = model.train(batch[0], batch[1])
		else:
			_, l = model.train2(batch[0], batch[1])
		all_loss.append(l)

		if batch_no%10 == 0:
			print("testing batch : ", batch_no)
			batch = mnist.test.images
			y_predict = model.predict(batch)
			acc_now = accuracy(mnist.test.labels, y_predict)
			all_accs.append(acc_now)
			print("accuracy", acc_now)

	return all_accs, all_loss
