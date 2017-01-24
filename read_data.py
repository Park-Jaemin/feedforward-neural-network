import os
from urllib.request import urlretrieve
import gzip
import autograd.numpy as np
import struct
import array


def download(url, filename):
    """
    Description: Download file if don't exist
    Input_1 (str): url where download from
    Input_2 (str): file name to download
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def parse_labels(filename):
    """
    Description: parse lables np_array from file
    Input_1 (str): file name where parse from
    Output_1 (np_array): lables
    """
    with gzip.open(filename, 'rb') as fh:
        magic, num_data = struct.unpack(">II", fh.read(8))
        return np.array(array.array("B", fh.read()), dtype=np.uint8)


def parse_images(filename):
    """
    Description: parse images np_array from file
    Input_1 (str): file name where parse from
    Output_1 (np_array): images
    """
    with gzip.open(filename, 'rb') as fh:
        magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
        return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)


def convert_to_one_hot(train_labels):
    y_ = np.zeros((60000, 10))
    y_[np.arange(60000), train_labels] = 1
