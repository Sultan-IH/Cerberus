import tensorflow as tf
import numpy as np


class Network(object):
    def __init__(self, layers):
        self.sess = tf.InteractiveSession()
        self.layers = layers
