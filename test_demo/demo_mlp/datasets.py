import gzip
import pickle
import sys
import struct
import tarfile

import numpy as np

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def mnist(path, one_hot=False):
    """
    return: train_set, valid_set, test_set
    train_set size: (50000, 784), (50000,)
    valid_set size: (10000, 784), (10000,)
    test_set size: (10000, 784), (10000,)
    feature: numerical in range [0, 1]
    target: categorical from 0 to 9
    """
    
    # load the dataset
    with gzip.open(path, "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    if one_hot:
        train_set = (train_set[0], get_one_hot(train_set[1], 10))
        valid_set = (valid_set[0], get_one_hot(valid_set[1], 10))
        test_set = (test_set[0], get_one_hot(test_set[1], 10))

    return train_set, valid_set, test_set