import tensorflow as tf
import numpy as np
import random as rn
from tensorflow.python.client import device_lib

"""
Does all of the dirty work: saving, training using a GPU
layers: a list of layers that would comprise the network

engine: an implementation of a learning algorithm used
"""
"""
LAYERS: MUST BE PASSED IN AS A LIST OF INITTIATED LAYERS
every layer must have a get_op() method that takes an op from a previous layer

"""
"""
COST:
"""


class NetworkConstructor():
    def __init__(self, Train_data, Test_data, batch_size,save_path ):
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(dtype=tf.float32)
        self.y = tf.placeholder(dtype=tf.float32)
        self.train_data = list(self.chunky(Train_data, batch_size))
        self.test_data = list(self.chunky(Test_data, batch_size))  # probably not tho
        self.params = []
        self.save_path = save_path

    def fit_layers(self, layers):
        """creates a graph depending on what layers have been fed to it."""

        for layer in layers:
            if layers.index(layer) == 0:
                op = layer.get_op(self.x)

            # should pass in the op from the previous layer to the next layer
            self.params.append(layer.get_params())
            op = layer.get_op(op)

        self.compute_op = op

    def fit_engine(self, engine):
        # might be unstable because passing in an empty class
        engine.cost_class.__init__(engine.cost_class, self.params)
        self.engine = engine

    def construct_and_train(self):
        # TODO: when reaches a peak accuracy with validation data save the netwrok
        # TODO: implement dropout
        # TODO: should the training be done in an engine or not

        self.sess.run(tf.global_variables_initializer())
        train_op = self.engine.get_train_op()
        batches = rn.shuffle(self.train_data)
        for e in range(self.engine.epochs):
            for b in batches:
                train_op.run(feed_dict={self.x: b[0], self.y: b[1]})

        Y = self.engine.propagate(_dict={self.x: self.train_data}, compute_op=self.compute_op)

        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(self.test_data[1], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.add_to_collection("accuracy",accuracy)
        self.save_full_graph(self.sess,self.save_path)




    @staticmethod
    def save_full_graph(sess, path):
        with sess.graph.as_default():
            saver = tf.train.Saver()
            saver.save(sess, path, meta_graph_suffix='meta', write_meta_graph=True)
    @staticmethod
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
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
