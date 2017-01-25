import tensorflow as tf
from Vengine.Layers.BaseLayerClass import Layer
# TODO: implement dropout

class DenseLayer(Layer):
    def __init__(self, shape):
        self.Weights = self.weight_variable(shape=shape)
        self.Biases = self.bias_variable([shape[1]])
        self.params = [self.Weights,self.Biases]

    def get_op(self, x):
        self.op = tf.nn.relu(tf.add(tf.matmul(x, self.Weights), self.Biases))
        return self.op
