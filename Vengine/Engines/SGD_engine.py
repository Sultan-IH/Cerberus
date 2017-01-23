import tensorflow as tf
import numpy as np
from Vengine.Engines.BaseEngine import Engine


class SGD_engine(Engine):
    """minimise the cost function ( provided in the __init__ method ) using stochastic gradient descent by backpropagation """

    def get_train_op(self, lr):
        """minimise using the cost derivative"""
        cost_fn = self.cost_class.cost_op()
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost_fn)
        return train_step


