import numpy as np
import tensorflow as tf
from face_net.image_processing import get_img_data
import pickle

imgdir = './face_datasets/'
train_datset = get_img_data(imgdir, sub_dir="Train_Data/")
print(train_datset.shape)


# test_dataset = get_img_data(imgdir, "Test_Data/", _save=True) TODO: get a test data set
# sess = tf.InteractiveSession()

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
