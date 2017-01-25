import tensorflow as tf
from tensorflow.python.client import device_lib
import random as rn


def train(net, epochs, Train_data, Test_data, batch_size):


    net.sess.run(tf.global_variables_initializer())
    train_op = net.engine.get_train_op()
    batches = rn.shuffle( Train_data)
    # TODO: training should be done here
    for e in range(epochs):
        for b in batches:
            train_op.run(feed_dict={net.x: b[0], net.y: b[1]})
            # TODO: when reaches a peak accuracy with validation data save the netwrok

    Y = net.compute_op.eval(_dict={net.x: Test_data})

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(net.test_data[1], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.add_to_collection("accuracy", accuracy)



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
