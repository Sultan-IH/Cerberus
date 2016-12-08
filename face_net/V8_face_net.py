import numpy as np
import tensorflow as tf
from face_net.image_processing import get_img_data
import pickle
imgdir = './face_datasets/'
# train_datset = get_img_data(imgdir, "Train_Data/", _save=True)
# # test_dataset = get_img_data(imgdir, "Test_Data/", _save=True) TODO: get a test data set
#
# sess = tf.InteractiveSession()
with open(imgdir+"Train_Data/dataset.pickle","rb+") as f:
    dataset = pickle.load(f)
print(dataset)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
