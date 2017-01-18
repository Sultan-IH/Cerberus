import tensorflow as tf
import numpy as np


class NetworkConstructor():
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(dtype=tf.float32)
        self.y = tf.placeholder(dtype=tf.float32)

    def fit_layers(self, layers):
        """creates a graph depending on what layers have been fed to it."""
        ops = []
        for l in layers:
            ops.append(l.return_op())


    def fit_engine(self, engine):
        self.train_op = engine.train_op()
        print("fitted engine")
        # TODO engine train

    def construct_and_train(self):
        self.sess.run(tf.global_variables_initializer())
        """
        Train using engine.train
        save the graph if the resutls are best in a special directory
        :return:
        """