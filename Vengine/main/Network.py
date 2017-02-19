import tensorflow as tf
import numpy as np
import random as rn

"""
Does all of the dirty work: saving, training using a GPU

LAYERS: MUST BE PASSED IN AS A LIST OF INITIATED LAYERS
every layer must have a get_op() method that takes an op from a previous layer

COST: MUST BE AN UNINITIATED COST CLASS
"""


class Network():
    """Initialises a graph """

    def __init__(self, engine, layers):
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(dtype=tf.float32)
        self.y = tf.placeholder(dtype=tf.float32)
        self.params = []
        for layer in layers:
            index = layers.index(layer)
            print("Layer: {0}".format(layer))

            if index == 0:
                op = layer.get_op(self.x)
            else:
                print("Current layer dims: {0} and previous layer dims: {1}".format(layer.dims, layers[index - 1].dims))
                if layer.dims != layers[index - 1].dims and layer.shape is not None:
                    # reshaping the activations
                    print("Reshaping; {0}".format(layer))
                    op = tf.reshape(layers[index - 1].op, [-1, layer.shape[0]])

                op = layer.get_op(op)

            # should pass in the op from the previous layer to the next layer
            try:
                self.params.append(layer.params)
            except:
                print("Encountered a Pooling layer")

        self.compute_op = op
        engine.cost_class = engine.cost_class(self.params)  # Initiating cost class
        self.train_op = engine.get_train_op(self.compute_op, self.y)
        self.sess.run(tf.global_variables_initializer())
        tf.add_to_collection("train_op", self.train_op)
        tf.add_to_collection("compute_op", self.compute_op)
        tf.add_to_collection("placeholders", [self.x, self.y])
