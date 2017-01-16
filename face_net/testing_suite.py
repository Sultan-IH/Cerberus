from face_net._processing import det_face_one
import tensorflow as tf

BATCH_SIZE = 20
people = ['Armaan', 'Jessica', 'Micah', 'Tal']
imgdir = './face_datasets/'
train_dir = "Train_data/"
test_dir = "Test_Data"
var = False

#
# # Creates a graph.
# with tf.device('/gpu:0'):
#   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#   c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))
print(tf.__file__)