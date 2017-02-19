import tensorflow as tf
from Vengine.Layers.BaseLayerClass import Layer


class ConvLayer(Layer):

    def get_op(self, x):
        """Should return its op"""
        self.op = tf.nn.relu(self.conv2d(x, self.Weights) + self.Biases)
        return self.op

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
