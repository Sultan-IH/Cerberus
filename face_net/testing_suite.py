import face_net._processing as pr
import cv2
import numpy as np

imgdir = './face_datasets/'
train_dir = "Train_data/"
test_dir = "Test_Data"
#pr.get_img_data(imgdir, train_dir)
raw_lables = pr._load(img_dir=imgdir, sub_dir=train_dir, file="labels")
raw_imgs = pr._load(img_dir=imgdir, sub_dir=train_dir, file="imgs")
print(type(raw_imgs[45]))
print(len(raw_imgs))

cv2.namedWindow("preview")
cv2.imshow('preview', raw_imgs[123])  # they are all in a list

cv2.waitKey(0)
cv2.destroyAllWindows()
