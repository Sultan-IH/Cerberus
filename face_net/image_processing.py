from scipy import misc
import os
import numpy as np

"""
Goal: produce training and testing data sets.
Entries should be in the form of: (image_data,label)
Sub-goals:
Iterate over every single file in each of the directories creating tuples. Add them to the data set.
"""


def get_img_data(img_dir, sub_dir, _save):
    imgs = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int)
    img_dir += sub_dir
    people = os.listdir(img_dir)
    people.pop(0)  # get rid of the .DS_Store
    try:
        people.remove('dataset.npy')  # ignore the existing dataset
    except ValueError:
        pass

    for person in people:
        faces = os.listdir(img_dir + '/' + person)
        faces.pop(0)  # get rid of the .DS_Store
        for face in faces:
            processed_face = misc.imread(img_dir + person + "/" + face, flatten=True)
            imgs = np.append(imgs, processed_face) # TODO: levae these as lists and convert them just before saving
            labels = np.append(labels, person)
    assert len(imgs) == len(labels)
    if _save:
        with open(img_dir + "dataset.npy", "wb") as f:
            print('Dumping the data set....')
            print("Length of first instance: {0}".format(len(imgs[0])))
            np.save(np.array(imgs, labels), f) # concatenate the list beforehand and convert it to numpy array here
    return imgs, labels


test_imgs, test_labels = get_img_data('./face_datasets/', "Train_data/", _save=True)
