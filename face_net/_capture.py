import cv2

"""Have to have a standartised image dimensions and then magnify and diminish frames accordingly"""
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
WIDTH = 400
HEIGHT = 400

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        just_face = frame[y:(y + w), x:(x + h)]
    try:
        resized_image = cv2.resize(just_face, (WIDTH, HEIGHT))
        cv2.imshow("preview", resized_image)
    except NameError:
        pass

    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
