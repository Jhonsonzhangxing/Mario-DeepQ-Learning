import tensorflow as tf
import numpy as np
from collections import deque

# {A, left, right}
ACTIONS = 3
PRESS_THRESHOLD = 0.5
GAMMA = 0.99
OBSERVE = 100
EXPLORE = 20000
FINAL_EPSILON = 0.001
INITIAL_EPSILON = 0.99
REPLAY_MEMORY_SIZE = 50000
BATCH = 32
FRAME_PER_ACTION = 1

class Model:
	def __init__(self):
		def conv_layer(x, conv, stride = 1):
			return tf.nn.conv2d(x, conv, [1, stride, stride, 1], padding = 'SAME')
		
		def pooling(x, k = 2, stride = 2):
			return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, stride, stride, 1], padding = 'SAME')

		self.middle_game = False
		self.memory = deque()
		self.initial_stack_images = np.zeros((80, 80, 4))
		self.X = tf.placeholder("float", [None, 80, 80, 4])
		self.actions = tf.placeholder("float", [None, ACTIONS])
		self.Y = tf.placeholder("float", [None])

		w_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.1))
		b_conv1 = tf.Variable(tf.truncated_normal([32], stddev = 0.01))
		self.test = tf.nn.relu(conv_layer(self.X, w_conv1) + b_conv1)
		conv1 = tf.nn.relu(conv_layer(self.X, w_conv1, stride = 4) + b_conv1)
		pooling1 = pooling(conv1)

		w_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.1))
		b_conv2 = tf.Variable(tf.truncated_normal([64], stddev = 0.01))
		conv2 = tf.nn.relu(conv_layer(pooling1, w_conv2, stride = 2) + b_conv2)

		w_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.1))
		b_conv3 = tf.Variable(tf.truncated_normal([64], stddev = 0.01))
		conv3 = tf.nn.relu(conv_layer(conv2, w_conv3) + b_conv3)

		conv3 = tf.reshape(conv3, [-1, 1600])
		w_fc1 = tf.Variable(tf.truncated_normal([1600, 512], stddev = 0.1))
		b_fc1 = tf.Variable(tf.truncated_normal([512], stddev = 0.01))
		fc_512 = tf.nn.relu(tf.matmul(conv3, w_fc1) + b_fc1)

		w_fc2 = tf.Variable(tf.truncated_normal([512, ACTIONS], stddev = 0.1))
		b_fc2 = tf.Variable(tf.truncated_normal([ACTIONS], stddev = 0.01))
		self.logits = tf.matmul(fc_512, w_fc2) + b_fc2

		readout_action = tf.reduce_sum(tf.multiply(self.logits, self.actions), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.Y - readout_action))
		self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.cost)








