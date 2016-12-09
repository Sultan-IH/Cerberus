from scipy import ndimage, misc
import os
import numpy as np
import pickle

"""
Goal: produce training and testing data sets.
Entries should be in the form of: (image_data,label)
Sub-goals:
Iterate over every single file in each of the directories creating tuples. Add them to the data set.
"""
imgdir = './face_datasets/'


def get_img_data(img_dir, sub_dir, _save=False):
    dataset = []
    img_dir += sub_dir
    people = os.listdir(img_dir)
    people.pop(0)  # get rid of the .DS_Store
    people.remove('dataset.pickle')  # ignore the existing dataset

    for person in people:
        faces = os.listdir(img_dir + '/' + person)
        faces.pop(0)  # get rid of the .DS_Store
        for face in faces:
            processed_face = misc.imread(img_dir + person + "/" + face, flatten=True)
            _set = (processed_face, person)
            dataset.append(_set)

    if _save:
        with open(img_dir + "dataset.pickle", "wb+") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('Dumping the data set....')
    return dataset

with open(imgdir + "Train_Data/dataset.pickle", "rb+") as f:
    data_set = pickle.load(f)
print(data_set.shape)