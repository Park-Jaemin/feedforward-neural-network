import gzip
import struct
import array
import autograd.numpy as np


def parse_labels(filename):
    with gzip.open(filename, 'rb') as fh:
        magic, num_data = struct.unpack(">II", fh.read(8))
        return np.array(array.array("B", fh.read()), dtype=np.uint8)


def parse_images(filename):
    with gzip.open(filename, 'rb') as fh:
        magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
        x = np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows * cols)
        x = np.divide(x, 255)
        return np.append(x, np.ones((len(x), 1)), axis=1)


def convert2onehot(train_labels):
    # convert to one-hot vector
    y_ = np.zeros((len(train_labels), 10))
    y_[np.arange(len(train_labels)), train_labels] = 1
    return y_
