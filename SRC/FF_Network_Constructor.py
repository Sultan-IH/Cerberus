import tensorflow as tf
import numpy as np

"""Does all of the dirty work: saving, training using a GPU
layers:
engine:
"""


class NetworkConstructor():
    def __init__(self, some_data):
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(dtype=tf.float32)
        self.y = tf.placeholder(dtype=tf.float32)
        self.input_data = some_data
        self.ops = []

    def fit_layers(self, layers):
        """creates a graph depending on what layers have been fed to it."""

        for l in layers:
            self.ops.append(l.return_op())

    def fit_engine(self, engine):
        self.engine = engine
        print("fitted engine")
        # TODO engine train

    def construct_and_train(self):
        # TODO: train using a range of GPUs
        self.sess.run(tf.global_variables_initializer())
        train_op = self.engine.train_op()
        batches = [self.input_data]
        for e in range(self.engine.epochs):
            for b in batches:
                train_op.run(feed_dict={self.x: b[0], self.y: b[1]})
            # evaluate
        # final test data check
        # saving the model
