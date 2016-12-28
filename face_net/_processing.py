# from scipy import misc
import os
import numpy as np
import random
import cv2

WIDTH = 400
HEIGHT = 450

# TODO: make them things one hot vectors
"""
Goal: produce training and testing data sets.
Entries should be in the form of: (image_data,label)
Sub-goals:
Iterate over every single file in each of the directories creating tuples. Add them to the data set.
"""


# Convert to grayscale

def get_img_data(img_dir, sub_dir, _ret=False):
    imgs = []
    labels = []
    img_dir += sub_dir
    people = os.listdir(img_dir)
    people.remove(".DS_Store")  # get rid of the .DS_Store
    try:
        people.remove('imgs')  # ignore the existing dataset
    except ValueError:
        pass
    try:
        people.remove('labels')  # ignore the existing dataset
    except ValueError:
        pass

    for person in people:
        print(person)
        faces = os.listdir(img_dir + '/' + person)
        try:
            faces.remove('.DS_Store')
        except:
            pass
        for face in faces:
            processed_face = cv2.imread(img_dir + person + "/" + face, 0)
            det = det_face_one(processed_face, 1.3)
            imgs.append(det)
            labels.append(people.index(person))
    assert len(imgs) == len(labels)

    print('Dumping the data set....')
    print('Removing None instances...')
    imgs = [x for x in imgs if x is not None]
    labels = [x for x in labels if x is not None]
    with open(img_dir + "imgs", "wb") as f:
        np.save(f, np.asarray(imgs))
    with open(img_dir + "labels", "wb") as f:
        np.save(f, np.asarray(labels))
    if _ret:
        return imgs, labels


def _load(img_dir, sub_dir, file):
    with open(img_dir + sub_dir + file, "rb") as f:
        dataset = np.load(f)
    return dataset


# simultaneous shuffle

def sim_shuffle(list1, list2):
    list1_shuf = []
    list2_shuf = []
    assert len(list1) == len(list2)
    indexes = list(range(len(list1)))
    random.shuffle(indexes)
    for i in indexes:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])
    return np.array(list1_shuf), np.array(list2_shuf)


def chunky(arr, size):  # TODO: need to convert to a list
    for l in range(0, len(arr), size):
        yield arr[l:l + size]


# arr = np.arange(9)
# _arr = chunky(arr, 2)
# print(list(_arr))

def det_face_one(img, scl):
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scl, 5)
    final_image = None
    for (x, y, w, h) in faces:
        just_face = img[y:(y + w), x:(x + h)]
        final_image = cv2.resize(just_face, (WIDTH, HEIGHT))

    return final_image
