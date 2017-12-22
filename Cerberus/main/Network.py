import tensorflow as tf
import numpy as np
import random as rn

"""
IDEAS:
multi gpu train
retraining particular layers(transfer training)

WHAT IS A NETWORK INSTANCE?:
params, placeholders
compute op
train op

"""


class Network():
    """Initialises a graph """

    def __init__(self, engine=None, layers=None, load_dict=None, name=None, log=None):
        if load_dict:
            self.load_through_dict(load_dict)

        else:
            self.name = name if name is not None else "Untitled_ML_model"
            self.sess = tf.InteractiveSession()
            self.x = tf.placeholder(dtype=tf.float32)
            self.y = tf.placeholder(dtype=tf.float32)
            self.params = []
            if layers is not None:
                self.layers = layers
                for layer in layers:
                    index = layers.index(layer)
                    print("Layer: {0}".format(layer))

                    if index == 0:
                        self.compute_op = layer.get_op(self.x)
                    else:
                        self.add_layer(layer, index)
                tf.add_to_collection("params", self.params)
            if engine is not None:
                self.fit_engine(engine)
            tf.add_to_collection("placeholders", [self.x, self.y])

    def add_layer(self, layer, index):
        """
        TODO: CALLING THIS METHOD OUTSIDE DOESNT WORK, the shapes will be different
        unless index == 0
            adds another layer to the stack

        :param layer: Initialised layer with a shape and dimensions
        :param index: index of the layer in the layers stack
        :return: None (just modifies the Network object)
        """
        if index == 0:
            self.compute_op = layer.get_op(self.x)
        else:
            print(
                "Current layer dims: {0} and previous layer dims: {1}".format(layer.dims, self.layers[index - 1].dims))
            if layer.dims != self.layers[index - 1].dims and layer.shape is not None:
                # reshaping the activations
                print("Reshaping; {0} index {1}".format(layer, index))
                reshaped_op = tf.reshape(self.layers[index - 1].op, [-1, layer.shape[0]])
                self.compute_op = layer.get_op(reshaped_op)
            else:
                self.compute_op = layer.get_op(self.compute_op)

        # should pass in the op from the previous layer tos the next layer
        try:
            self.params.append(layer.params)
        except:
            print("Encountered a Pooling layer")

        tf.add_to_collection("compute_op", self.compute_op)  # when accessing it get the item at [-1]index

    def fit_engine(self, engine):
        engine.cost_class = engine.cost_class(self.params)  # Initiating cost class
        self.train_op = engine.get_train_op(self.compute_op, self.y)
        # TODO: log device placement
        self.sess.run(
            tf.global_variables_initializer())
        tf.add_to_collection("train_op", self.train_op)

    def load_through_dict(self, load_dict):
        self.sess = load_dict["sess"]
        self.params = load_dict["params"]
        self.x = load_dict["placeholders"][0]
        self.y = load_dict["placeholders"][1]
        self.compute_op = load_dict["train_op"]
        self.train_op = load_dict["train_op"]
