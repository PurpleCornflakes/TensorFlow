import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def baby0():
	node1 = tf.constant(3.0, dtype = tf.float32)
	node2 = tf.constant(4.0, dtype = tf.float32)
	print([node1, node2])

	# create a session to run
	sess = tf.Session()
	print(sess.run([node1,node2]))

def baby1():
	# constuct the static framework
	a = tf.placeholder(dtype = tf.float32)
	b = tf.placeholder(dtype = tf.float32)

	# operations are also nodes
	c = a + b
	d = c*3

	# start a session
	sess = tf.Session()
	print(sess.run(c,{a:3, b:4}))

def linear_regression(x_train, y_train):
	# parameters: not initialized yet
	W = tf.Variable([0.3], tf.float32,name = "xiaoll")
	b = tf.Variable([-0.3], tf.float32)
	# input and output
	x = tf.placeholder(tf.float32, name = "xiaocc")
	y = tf.add(W*x, b)

	# Desired value
	d = tf.placeholder(tf.float32) 

	# loss
	loss = tf.reduce_sum(tf.square(y - d))
	
	# optimizer	
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
	train = optimizer.minimize(loss)

	# start a session and initialize
	sess = tf.Session()
	# a handle to initialize all global variables
	init = tf.global_variables_initializer()
	sess.run(init)
	# training loop
	for i in range(1000):
		sess.run(train, {x: x_train, d: y_train})
	# final accuracy
	print(sess.run(loss,{x: x_train, d: y_train}))
	curr_W, curr_b, curr_loss = sess.run([W,b,loss], {x: x_train, d: y_train})
	print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


	def test_model(x_test = x_train, y_test= y_train):
		# fixW is a TensorNode to change W to trained value
		fixW = tf.assign(W, [1])
		fixb = tf.assign(b, [-1])
		sess.run([fixW, fixb])
		print(sess.run(loss, {x: x_test, d: y_test}))
	test_model()

if __name__ == "__main__":
	x_train = [1,2,3,4]
	y_train = [0,-1,-2,-3]
	linear_regression(x_train, y_train)