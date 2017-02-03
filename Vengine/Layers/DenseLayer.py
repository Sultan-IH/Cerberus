import tensorflow as tf
from Vengine.Layers.BaseLayerClass import Layer

# TODO: finish dropout
class DenseLayer(Layer):
    def __init__(self, shape, drop_prob=False):
        self.Weights = self.weight_variable(shape=shape)
        self.Biases = self.bias_variable([shape[1]])
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
