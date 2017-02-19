import tensorflow as tf
from tensorflow.python.client import device_lib
import random as rn

import numpy as np

"""
TRAIN_DATA list of two
"""


def train(net, epochs, data_sets, batch_size, log):
    train_op = net.train_op

    image_batches = list(chunky(data_sets["Train_data"][0], batch_size))
    lable_batches = list(chunky(data_sets["Train_data"][1], batch_size))
    batches = list(zip(image_batches, lable_batches))
    rn.shuffle(batches)
    Z = tf.placeholder(dtype=tf.float32)
    correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(data_sets["Test_data"][1], 1))
    accuracy = tf.mul(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)), 100)

    for e in range(epochs):

        for b in batches:
            train_op.run(feed_dict={net.x: b[0], net.y: b[1]})

        if data_sets["Validation_data"]:
            print("Epoch: {0}; accuracy: {1}".format(e, accuracy.eval(
                feed_dict={Z: net.compute_op.eval(feed_dict={net.x: data_sets["Validation_data"][0]}),
                           net.y: data_sets["Validation_data"][1]})))
        else:
            print("Epoch {0} finished; accuracy: {1}".format(e, accuracy.eval(
                feed_dict={Z: net.compute_op.eval(feed_dict={net.x: data_sets["Test_data"][0]}),
                           net.y: data_sets["Test_data"][1]})))

    Y = net.compute_op.eval(_dict={net.x: data_sets["Test_data"][0]})
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(data_sets["Test_data"][1], 1))  # should be a placeholder
    accuracy = tf.mul(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)), 100)
    final = accuracy.eval(feed_dict={net.x: data_sets["Test_data"][0], net.y: data_sets["Test_data"][1]})
    print("Final accuracy: {0}".format(final))
    tf.add_to_collection("Final_Accuracy", final)


"""Helper methods"""


def chunky(arr, size):
    for l in range(0, len(arr), size):
        yield arr[l:l + size]


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


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
