# TODO: implement dropout
import tensorflow as tf

class Layer():
    """provide base functionality for every layer"""
    @staticmethod
    def dropout(op, keep_prob):
        return tf.nn.dropout(op,keep_prob=keep_prob)
    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)