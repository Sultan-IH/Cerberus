# from scipy import misc
import os
import numpy as np
import random
import cv2

# TODO: change from using misc to cv2 and face detection
"""
Goal: produce training and testing data sets.
Entries should be in the form of: (image_data,label)
Sub-goals:
Iterate over every single file in each of the directories creating tuples. Add them to the data set.
"""


def get_img_data(img_dir, sub_dir, _ret=False):
    imgs = []
    labels = []
    img_dir += sub_dir
    people = os.listdir(img_dir)
    # people.pop(0)  # get rid of the .DS_Store
    try:
        people.remove('imgs')  # ignore the existing dataset
    except ValueError:
        pass
    try:
        people.remove('labels')  # ignore the existing dataset
    except ValueError:
        pass

    for person in people:
        faces = os.listdir(img_dir + '/' + person)
        faces.pop(0)  # get rid of the .DS_Store
        for face in faces:
            processed_face = misc.imread(img_dir + person + "/" + face, flatten=True)
            imgs.append(processed_face)
            labels.append(people.index(person))
    assert len(imgs) == len(labels)

    print('Dumping the data set....')
    print("Length of first instance: {0}".format(len(imgs[0])))
    with open(img_dir + "imgs", "wb") as f:
        np.save(f, np.array(imgs))
    with open(img_dir + "lables", "wb") as f:
        np.save(f, np.array(labels))
    if _ret:
        return imgs, labels


def _load(img_dir, sub_dir, file):
    with open(img_dir + sub_dir + file, "rb") as f:
        dataset = np.load(f)
    return dataset


def _shuffle(list1, list2):
    list1_shuf = []
    list2_shuf = []
    assert len(list1) == len(list2)
    indexes = list(range(len(list1)))
    random.shuffle(indexes)
    for i in indexes:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])
    return np.array(list1_shuf), np.array(list2_shuf)
