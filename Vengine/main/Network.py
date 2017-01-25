import tensorflow as tf
import numpy as np
import random as rn

"""
Does all of the dirty work: saving, training using a GPU
layers: a list of layers that would comprise the network

engine: an implementation of a learning algorithm used
"""
"""
LAYERS: MUST BE PASSED IN AS A LIST OF INITIATED LAYERS
every layer must have a get_op() method that takes an op from a previous layer

"""
"""
COST:
"""


class Network():

    def __init__(self, engine, layers ):
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(dtype=tf.float32)
        self.y = tf.placeholder(dtype=tf.float32)
        self.params = []
        for layer in layers:
            if layers.index(layer) == 0:
                op = layer.get_op(self.x)

            # should pass in the op from the previous layer to the next layer
            self.params.append(layer.get_params())
            op = layer.get_op(op)
        self.compute_op = op

        engine.cost_class.__init__(engine.cost_class, self.params)
        self.train_op = engine.get_train_op()
        tf.add_to_collection("train_op", self.train_op)
        tf.add_to_collection("compute_op", self.compute_op)
        tf.add_to_collection("placeholders", [self.x, self.y])





