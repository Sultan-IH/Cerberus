import tensorflow as tf
from tensorflow.python.client import device_lib
import random as rn

import numpy as np

"""
TRAIN_DATA list of two

"""


def train(net, epochs, data_sets, batch_size):
    print("train() called")
    train_op = net.train_op

    image_batches = [data_sets["Train_data"][0][k:k + batch_size] for k in
                     range(0, len(data_sets["Train_data"][0]), batch_size)]
    label_batches = [data_sets["Train_data"][1][k:k + batch_size] for k in
                     range(0, len(data_sets["Train_data"][1]), batch_size)]

    batches = list(zip(image_batches, label_batches))
    rn.shuffle(batches)

    if data_sets["Validation_data"]:
        Z, accuracy = make_accuracy_op(data_sets["Validation_data"][1])

    else:
        Z, accuracy = make_accuracy_op(data_sets["Test_data"][1])

    data_set = data_sets["Test_data"][1] if data_sets["Validation_data"] is None else data_sets["Validation_data"][1]
    GPUs = get_available_gpus()

    if GPUs is not []:
        for GPU in GPUs:
            with tf.device(GPU):
                with some_scope:
                    """CALCULATE tower loss"""

    for e in range(epochs):

        for b in batches:
            train_op.run(feed_dict={net.x: b[0], net.y: b[1]})
        acc = accuracy.eval(feed_dict={Z: net.compute_op.eval(feed_dict={net.x: data_set})})
        print("Epoch: {0}; Accuracy: {1}".format(e, acc))
        tf.add_to_collection("Accuracies", acc)

    Y, final = make_accuracy_op(data_sets["Test_data"][1])
    final_acc = final.eval(feed_dict={Y: data_sets["Test_data"][0]})
    print("Training finished; final accuracy: {0}".format(final_acc))
    tf.add_to_collection("Accuracies", final_acc)


"""Helper methods"""


def make_accuracy_op(labels):
    Z = tf.placeholder(dtype=tf.float32)
    correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(labels, 1))
    accuracy_op = tf.mul(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)), 100)
    return Z, accuracy_op


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
