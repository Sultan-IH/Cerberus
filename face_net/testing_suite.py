import face_net._processing as pr
import cv2
import random as rn
import tensorflow as tf

BATCH_SIZE = 20
people = ['Armaan', 'Jessica', 'Micah', 'Tal']
imgdir = './face_datasets/'
train_dir = "Train_data/"
test_dir = "Test_Data"
