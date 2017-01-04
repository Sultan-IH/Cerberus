from face_net._processing import det_face_one
import cv2
import tensorflow as tf

BATCH_SIZE = 20
people = ['Armaan', 'Jessica', 'Micah', 'Tal']
imgdir = './face_datasets/'
train_dir = "Train_data/"
test_dir = "Test_Data"
face = cv2.imread(imgdir + train_dir + people[1] + '/' + 'IMG_3713.JPG', 0)
assert face is not None
assert det_face_one(face, 1.1) is not None
test_face = [det_face_one(face, 1.1)]
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
new_saver = tf.train.import_meta_graph('V8_face_net_4L_CNN.meta')
new_saver.restore(sess, 'V8_face_net_4L_CNN')

compute_op = tf.get_collection('compute_op')[0]
x = tf.get_collection('x_placeholder')[0]

print(sess.run(compute_op, feed_dict={x: test_face}))
