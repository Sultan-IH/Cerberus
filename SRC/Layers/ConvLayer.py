import tensorflow as tf
import numpy as np


class ConvLayer(object):

    def __init__(self, filter_size, x):
        """ initialise parameters fo the layer
        filter_size: tuple like (5,5,1,15) where 5 and 5 is the stride size, and 15 is the feature map
        x: must be a placeholder value
        """
        assert len(filter_size) == 3
        self.Weights = self.weight_variable(filter_size)
        self.Biases = self.bias_variable(filter_size[3])
        self.op = tf.nn.relu(self.conv2d(x, self.Weights) + self.Biases)

    def return_op(self):
        """Should return its operation as a graph"""
        return self.op

    def set_input(self):
        """feed through the input"""

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
