import tensorflow as tf
import numpy as np
from SRC.Engines.BaseEngine import Engine


class SGD_engine(Engine):
    """minimise the cost function ( provided in the __init__ method ) using stochastic gradient descent by backpropagation """

    def __init__(self, cost_class):
        """cost"""
        self.cost_class = cost_class

    def get_train_op(self, lr):
        """minimise using the cost derivative"""
        cost_fn = self.cost_class.cost_op()
        train_step = tf.train.AdamOptimizer(lr).minimize(cost_fn)
        return train_step

    def propagate(self, placeholder, x, compute_op):
        """compute the funal value when a placeholder is fed"""
        return compute_op.eval(feed_dict={placeholder: x})
