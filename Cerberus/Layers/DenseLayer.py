import tensorflow as tf
from Cerberus.Layers.BaseLayerClass import Layer

class DenseLayer(Layer):
    def __init__(self, shape, drop_prob=False):
        self.Weights = self.weight_variable(shape=shape)
        self.Biases = self.bias_variable([shape[1]])
        self.shape = shape
        self.dims = len(shape)
        if drop_prob:
            self.drop_prob = tf.placeholder(dtype=tf.float32)
        else:
            self.drop_prob = False

        self.params = [self.Weights, self.Biases]

    def get_op(self, x):
        self.op = tf.nn.relu(tf.add(tf.matmul(x, self.Weights), self.Biases))

        if self.drop_prob:
            self.op = self.dropout(self.op, keep_prob=self.drop_prob)

        return self.op
