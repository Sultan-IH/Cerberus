import tensorflow as tf
from Vengine.Layers.BaseLayerClass import Layer


class ConvLayer(Layer):
    def __init__(self, filter_size):
        """ initialise parameters fo the layer
        filter_size: tuple like (5,5,1,15) where 5 and 5 is the stride size, and 15 is the feature map
        x: must be a placeholder/tensor
        """

        assert len(filter_size) == 3

        self.Weights = self.weight_variable(filter_size)
        self.Biases = self.bias_variable(filter_size[3])
        self.params = [self.Weights, self.Biases]

    def get_op(self, x):
        """Should return its op"""
        self.op = tf.nn.relu(self.conv2d(x, self.Weights) + self.Biases)
        return self.op

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
