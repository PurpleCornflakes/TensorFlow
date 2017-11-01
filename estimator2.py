import tensorflow as tf
import numpy as np 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# defines what to predict, loss function, optimizer
def model_fn(features, labels, mode):
	# Build linear model and output
	W = tf.get_variable(name = "W", shape=[1], dtype=tf.float64)
	b = tf.get_variable("b", [1], tf.float64)
	# tf.get_variable(name = "xx")
	print(W)
	y = W*features['x'] + b

	# Loss subgraph
	loss = tf.reduce_sum(tf.square(y-labels))

	# Training subgraph
	global_step = tf.train.get_global_step()
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
	# train one step, global_step++
	train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

	# Spec connects subgraphs
	return tf.estimator.EstimatorSpec(
		mode = mode,
		predictions = y,
		loss = loss,
		train_op = train)

estimator = tf.estimator.Estimator(model_fn = model_fn)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# Train
estimator.train(input_fn=input_fn, steps=1000)
# Evaluate how well the model did
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)