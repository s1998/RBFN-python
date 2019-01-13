import numpy as np

class parameterWithAdamOpt:
	def __init__(self, obj, lr = 0.001):
		self.param = obj
		self.m = 0
		self.v = 0
		self.t = 0
		self.lr = lr
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.epsilon = 0.00000001

	def update(self, derivative):
		self.t += 1
		self.lr = self.lr * np.sqrt(1 - pow(self.beta2, self.t)) / (1 - pow(self.beta1, self.t))
		m_n = self.beta1 * self.m + (1 - self.beta1) * derivative
		v_n = self.beta2 * self.v + (1 - self.beta2) * derivative * derivative
		self.m = m_n
		self.v = v_n
		self.param = self.param - self.lr * self.m / (np.sqrt(self.v) + self.epsilon)
