from face_net._processing import det_face_one
import cv2
import tensorflow as tf

BATCH_SIZE = 20
people = ['Armaan', 'Jessica', 'Micah', 'Tal']
imgdir = './face_datasets/'
train_dir = "Train_data/"
test_dir = "Test_Data"
var = False


def reset():
    time.sleep(60)
    print(var)
    global var
    var = True


threading.Thread(target=reset).start()
