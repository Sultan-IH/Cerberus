import cv2
from face_net._processing import det_face_one, softmax
import tensorflow as tf
import threading
import time

# TODO: k-means clustering with faces. cool stuff
"""Have to have a standartised image dimensions and then magnify and diminish frames accordingly"""
cv2.namedWindow("face")
vc = cv2.VideoCapture(0)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
new_saver = tf.train.import_meta_graph('V8_face_net_4L_CNN.meta')
new_saver.restore(sess, 'V8_face_net_4L_CNN')

compute_op = tf.get_collection('compute_op')[0]
x = tf.get_collection('x_placeholder')[0]

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    face = det_face_one(frame, 1.3)
    if face is not None:
        print('TRYING TO RECOGNISE A FACE')
        cv2.imshow('face', face)
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        acts = compute_op.eval(feed_dict={x: [face]})
        probs = softmax(acts[0])
        print(probs)

    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
