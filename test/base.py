
import os
import csv
import shutil
import warnings
from collections import namedtuple
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext
import hashlib


from utils_for_dataload import Bunch


import numpy as np

from urllib.request import urlretrieve


def load_digits(n_class=10, return_X_y=False):
    module_path = dirname(__file__)
    data = np.load("./data.npy")
    target = np.load("./target.npy")
    with open("digits.rst") as f:
        descr = f.read()
    flat_data = data
    images = flat_data.view()

    if return_X_y:
        return flat_data, target

    return Bunch(data=data,
                 target=target,
                 target_names=np.array([1,2]),
                 images=images,
                 DESCR=descr)

