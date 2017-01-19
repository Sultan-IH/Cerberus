import tensorflow as tf
import numpy as np
import random as rn

"""
Does all of the dirty work: saving, training using a GPU
layers: a list of layers that would comprise the network
engine: an implementation of a learning algorithm used
"""


class NetworkConstructor():
    def __init__(self, data_set, batch_size):
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(dtype=tf.float32)
        self.y = tf.placeholder(dtype=tf.float32)
        self.train_data = self.chunky(data_set, batch_size)
        self.test_data = []

    def fit_layers(self, layers):
        """creates a graph depending on what layers have been fed to it."""
        op = layers[0].get_op(self.x)
        for l in layers:
            # should pass in the op from the previous layer to the next layer
            op = l.get_op(op)
        self.Y_ = op

    def fit_engine(self, engine):
        self.engine = engine
        print("fitted engine")

    def construct_and_train(self):
        # TODO: train using a range of GPUs
        # TODO: when reaches a peak accuracy with validation data save the netwrok
        # TODO: implement dropout

        self.sess.run(tf.global_variables_initializer())
        train_op = self.engine.get_train_op()
        batches = rn.shuffle(self.train_data)
        for e in range(self.engine.epochs):
            for b in batches:
                train_op.run(feed_dict={self.x: b[0], self.y: b[1]})
        self.engine.propagate().eval(feed_dict={self.x: self.train_data[0]})
        correct_prediction = tf.equal(tf.argmax(self.Y_, 1), tf.argmax(self.test_data[1], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # saving the model

    @staticmethod
    def chunky(arr, size):
        for l in range(0, len(arr), size):
            yield arr[l:l + size]

    @staticmethod
    def sim_shuffle(list1, list2):
        list1_shuf = []
        list2_shuf = []
        assert len(list1) == len(list2)
        indexes = list(range(len(list1)))
        rn.shuffle(indexes)
        for i in indexes:
            list1_shuf.append(list1[i])
            list2_shuf.append(list2[i])
        return np.array(list1_shuf), np.array(list2_shuf)
