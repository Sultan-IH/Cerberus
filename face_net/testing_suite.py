import face_net._processing as pr
import cv2
import random as rn

people = ['Armaan', 'Jessica', 'Micah', 'Tal']
imgdir = './face_datasets/'
train_dir = "Train_data/"
test_dir = "Test_Data"

# pr.get_img_data(imgdir, train_dir)

raw_lables = pr._load(img_dir=imgdir, sub_dir=train_dir, file="labels")
raw_imgs = pr._load(img_dir=imgdir, sub_dir=train_dir, file="imgs")

imgs, lables = pr.sim_shuffle(raw_imgs, raw_lables)


ind = rn.randint(0, len(imgs))
person = lables[ind]

cv2.namedWindow('preview')
cv2.imshow('preview', imgs[ind])  # they are all in a list
print(person)

cv2.waitKey(0)
cv2.destroyAllWindows()
# Everything works so far; need to adjust the scale factor