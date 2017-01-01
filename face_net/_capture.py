import cv2
from face_net._processing import det_face_one
import numpy as np

"""Have to have a standartised image dimensions and then magnify and diminish frames accordingly"""
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    face = det_face_one(frame, 1.3)
    try:
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        cv2.imshow("preview", face)
    except:
        pass

    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
# TODO: