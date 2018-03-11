from sklearn.cluster import KMeans
import numpy as np 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# __init__ function will make the computation graph
# a separate function will be used for training 
# a separate function will be used for testing




class rbfn_model_tf:
	def __init__(self, indim, num_classes, no_of_clusters, x, variabl_cc = True):
		
		# define variables
		self.indim = indim
		self.num_classes = num_classes
		self.no_of_clusters = no_of_clusters
		self.weights = tf.Variable(tf.random_uniform((no_of_clusters, num_classes)))
		self.beta = tf.Variable(-1.0)
		
		# get clusters
		if variabl_cc:
			self.cluster_centres = tf.Variable(np.random.permutation(x)[:self.no_of_clusters, :].reshape(1, self.no_of_clusters, self.indim))
		else:
			self.cluster_centres = tf.constant(np.random.permutation(x)[:self.no_of_clusters, :].reshape(1, self.no_of_clusters, self.indim))

		# define placeholders and define computation graph
		self.x_input = tf.placeholder(tf.float32, [None, indim])
		# print("x", self.x_input.shape)
		self.x_input_r = tf.reshape(self.x_input, (-1, 1, 784))
		# print("x_r", self.x_input_r.shape)
		self.x_input_d = self.x_input_r - self.cluster_centres
		# print("x_d", self.x_input_d.shape)
		self.x_input_d_2 = tf.square(self.x_input_d)
		# print("x_d_2", self.x_input_d.shape)
		self.x_input_s = tf.reduce_sum(self.x_input_d_2, axis = 2) 
		# print("x_s", self.x_input_s.shape)

		self.x_input_g = tf.exp(self.beta * self.x_input_s)
		# print("x_g", self.x_input_g.shape)
		# self.y_input = tf.placeholder(tf.uint8, [None])
		# self.y_input_o = tf.one_hot(indices = self.y_input,
		# 	depth = self.num_classes,
		# 	on_value = 1.0,
		# 	off_value = 0.0,
		# 	axis = -1)
		self.y_input = tf.placeholder(tf.uint8, [None, 10])
		self.y_input_o = self.y_input
		self.y_predicted = tf.matmul(self.x_input_g, self.weights)
		# print("y_p", self.y_predicted.shape)
		
		self.loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.y_predicted, labels = self.y_input_o)
		self.loss_t = tf.reduce_sum(self.loss)

		self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
		self.trainer = self.optimizer.minimize(self.loss_t)
		
		self.optimizer2 = tf.train.AdamOptimizer(learning_rate = 0.001)
		self.trainer2 = self.optimizer2.minimize(self.loss_t)

		self.sess = tf.Session()
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)

	def train(self, x, y):
		result = self.sess.run(self.trainer, feed_dict={self.x_input: x, self.y_input:y})

	def train2(self, x, y):
		result = self.sess.run(self.trainer2, feed_dict={self.x_input: x, self.y_input:y})

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

# initializing rbfn model
model = rbfn_model_tf(784, 10, 100, mnist.train.next_batch(5000)[0])
model2 = rbfn_model_tf(784, 10, 100, mnist.train.next_batch(5000)[0], False)
print("\n\nInitialized the model with random centres \n\n\n\n")

acc_now = 0
for batch_no in range(2000):
	batch = mnist.train.next_batch(500)
	# print(batch[0].shape)
	# print(batch[1].shape)
	if acc_now < 70:
		model.train(batch[0], batch[1])
	else:
		model.train2(batch[0], batch[1])
	
	if batch_no%14 == 0:
		print("testing batch : ", batch_no)
		batch = mnist.test.images
		y_predict = model.predict(batch)
		acc_now = accuracy(mnist.test.labels, y_predict)
		print("accuracy", acc_now)

print("\n\nVariable_cc is false now \n")
acc_now = 0
for batch_no in range(2000):
	batch = mnist.train.next_batch(500)
	# print(batch[0].shape)
	# print(batch[1].shape)
	if acc_now < 70:
		model2.train(batch[0], batch[1])
	else:
		model2.train2(batch[0], batch[1])
	
	if batch_no%14 == 0:
		print("testing batch : ", batch_no)
		batch = mnist.test.images
		y_predict = model2.predict(batch)
		acc_now = accuracy(mnist.test.labels, y_predict)
		print("accuracy", acc_now)

































