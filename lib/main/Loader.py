import tensorflow as tf
import numpy as np


class LoadNetwork(object):
    def __init__(self, path):
        """load path"""
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.import_meta_graph(path)
        self.saver.restore(self.sess, path)
