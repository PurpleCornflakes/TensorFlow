import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
# 55000 in mnist.train; 10000 in mnist.test; 5000 in mnist.validation
# one_hot: label is like [0,1,0,0,0,0,0,0,0,0]
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
assert(mnist.train.images.shape == (55000, 784))
assert(mnist.train.labels.shape == (55000, 10))

def draw_img(img_1D = mnist.train.images[0]):
	plt.matshow(img_1D.reshape(28,-1))
	plt.show()

# MLP structure: 784*500*10
tf.set_random_seed(222)
def init_weights(shape):
	weights = tf.random_normal(shape, stddev=0.3)
	return tf.Variable(weights)

def multilayer_percerptron(x_train = mnist.train.images, y_train = mnist.train.labels, num_epochs = 1):

	x_size = 784
	h1_size = 500
	predict_size = 10

	x = tf.placeholder(tf.float32, shape=[None, x_size], name='x') # None: can be any length
	W1 = init_weights([x_size, h1_size])
	# W1 = tf.Variable(tf.zeros([x_size, h1_size]), name='W1')
	b1 = tf.Variable(tf.zeros([1, h1_size]), name='b1')
	h1 = tf.nn.relu(tf.matmul(x, W1) + b1, name='h1')
	W2 = init_weights([h1_size, predict_size])
	# W2 = tf.Variable(tf.zeros([h1_size, predict_size]), name='W2')
	b2 = tf.Variable(tf.zeros([1, predict_size]), name='b2')
	predict = tf.nn.relu(tf.matmul(h1, W2) + b2, name='predict')

	y = tf.placeholder(tf.float32, shape=[1, predict_size], name='y')

	loss = tf.reduce_sum(tf.square(y - predict))
	optimizor = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
	train = optimizor.minimize(loss)
	init = tf.global_variables_initializer()
	# return train

# def train_model(train = train)
	sess = tf.Session()
	sess.run(init)
	N = 1000#x_train.shape[0]
	# max_W1 = np.zeros(N, dtype = np.float32)
	loss_list = np.zeros(N, dtype = np.float32)
	for epoch in range(num_epochs):
		for i in range(N):
			xx = x_train[i].reshape(1, -1).astype(np.float32)
			yy = y_train[i].reshape(1, -1).astype(np.float32)
			assert(xx.shape == (1, 784) and xx.dtype == "float32")
			assert(yy.shape == (1, 10) and yy.dtype == "float32")
			sess.run(train, {x: xx, y: yy})
			# max_W1[i] = np.max(sess.run([W1]))#, b1, W2, b2]))
			loss_list[i] = sess.run(loss, {x: xx, y: yy})
	final_W1, final_b1, final_W2, final_b2 = sess.run([W1, b1, W2, b2])
	# print(loss_list.shape)
	# plt.plot(loss_list,'o')
	# plt.show()
	def test_model(x_test = mnist.test.images, y_test = mnist.test.labels):
		fix_W1 = tf.assign(W1, final_W1)
		fix_b1 = tf.assign(b1, final_b1)
		fix_W2 = tf.assign(W2, final_W2)
		fix_b2 = tf.assign(b2, final_b2)
		sess.run([fix_W1, fix_b1, fix_W2, fix_b2])
		assert(len(y_test) == 10000)
		test_accuracy = np.mean([sess.run(predict, {x: x_test[i].reshape(1, -1).astype(np.float32)}) == y_test[i].reshape(1, -1).astype(np.float32) 
			for i in range(len(y_test))])
		print(test_accuracy)
	test_model()


if __name__ == "__main__":
	multilayer_percerptron()
	# draw_img()
