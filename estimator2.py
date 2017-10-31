import tensorflow as tf
import numpy as np 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# defines how to predict, loss function, optimizer
def model_fn(features, labels, mode):
	# Build linear model and output
	W = tf.get_variable("W", [1], tf.float64)
	b = tf.get_variable("b", [1], tf.float64)

	y = W*features['x']
