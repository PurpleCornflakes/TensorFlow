import tensorflow as tf
import numpy as np 
# to mute warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''
tf.estimator simplifies 
1. train/eval loops by numpy_input_fn:
2. evaluate model by estimator.evaluate(input_fn)
'''

feature_columns = [tf.feature_column.numeric_column("x", shape = [1])]
# many predefined classifiers and regressors 
estimator = tf.estimator.LinearRegressor(feature_columns = feature_columns)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

# feed data while training model
input_fn = tf.estimator.inputs.numpy_input_fn(
	{"x": x_train}, y_train, 
	batch_size = 4, num_epochs = None, 
	shuffle = True)

# feed x_train, y_train when evaluating model 
train_input_fn = tf.estimator.inputs.numpy_input_fn(
	{"x": x_train}, y_train,
	batch_size = 4, num_epochs = 1,
	shuffle = False)

# feed x_eval, y_eval when evaluating model 
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	{"x": x_eval}, y_eval,
	batch_size = 4, num_epochs = 1000,
	shuffle = False)

# invoke training
estimator.train(input_fn = input_fn, steps = 1000)

# evaluate how well the model
train_metrics = estimator.evaluate(input_fn = train_input_fn)
eval_metrics = estimator.evaluate(input_fn = eval_input_fn)

print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)

