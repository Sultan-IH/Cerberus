import tensorflow as tf
from Vengine.Layers.BaseLayerClass import Layer


class PoolLayer(Layer):
    def __init__(self, filter_size):
        self.Weights = self.weight_variable(filter_size)
        self.Biases = self.bias_variable(filter_size[3])
        self.params = [self.Weights, self.Biases]

    def get_op(self, x):
        """Should return its op"""
        self.op = self.max_pool_2x2(x)
        return self.op

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
