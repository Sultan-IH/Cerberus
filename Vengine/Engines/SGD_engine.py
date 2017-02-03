import tensorflow as tf
from Vengine.Engines.BaseEngine import Engine


class SGD_engine(Engine):
    """minimise the cost function ( provided in the __init__ method ) using stochastic gradient descent by backpropagation """

    def get_train_op(self, compute_op,labels):
        """minimise using the cost derivative"""
        print("get_train_op")
        cost_fn = self.cost_class.get_cost_fn(compute_op, labels, 1e-2)
        train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(cost_fn)
        return train_step
