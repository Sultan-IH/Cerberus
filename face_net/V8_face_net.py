import numpy as np
import tensorflow as tf
import face_net.image_processing as img
import sys
import random

imgdir = './face_datasets/'
train_dir = "Train_data/"
test_dir = "Test_Data/"

train_data = img.get_img_data(img_dir=imgdir, sub_dir=train_dir, _save=False)
# data is delivered in the form of a tuple like: (img_data,label)
print("Size of the train data: = {0}".format(sys.getsizeof(train_data)))
print("Length of the data set: {0}; Shape of each element: {1}".format(len(train_data), train_data[0][0].shape))
random.shuffle(train_data)
labels = [elm[1] for elm in train_data]  # TODO: make a simple sorting function
imgs = [elm[0] for elm in train_data]

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
