import face_net._processing as pr
import cv2
import random as rn
BATCH_SIZE = 20
people = ['Armaan', 'Jessica', 'Micah', 'Tal']
imgdir = './face_datasets/'
train_dir = "Train_data/"
test_dir = "Test_Data"

# pr.get_img_data(imgdir, train_dir)

raw_lables = pr._load(img_dir=imgdir, sub_dir=train_dir, file="labels")
raw_imgs = pr._load(img_dir=imgdir, sub_dir=train_dir, file="imgs")
test_image = raw_imgs[199]
test_label = raw_lables[11]
cv2.namedWindow('preview')
cv2.imshow('preview', test_image)  # they are all in a list
print(test_label)

cv2.waitKey(0)
cv2.destroyAllWindows()

# img_batches = list(pr.chunky(raw_imgs, BATCH_SIZE))
# lables_batches = list(pr.chunky(raw_lables, BATCH_SIZE))
#
# imgs, lables = pr.sim_shuffle(img_batches, lables_batches)
#
#
# ind = rn.randint(0, len(imgs))
# person = lables[ind][0]
#

# # Everything works so far; need to adjust the scale factor