import numpy as np
import matplotlib.pyplot as plt

import os

from keras.src.legacy.preprocessing.image import ImageDataGenerator
# from keras.src.models import model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.src.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Concatenate
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.summary import summary


import h5py

def print_structure(weight_file_path):
    """ Prints out the structure of HDF5 file, including number of layers.

    Args:
        weight_file_path (str): Path to the file to analyze
    """
    f = h5py.File(weight_file_path, 'r')  # Open in read-only mode
    try:
        # Check for existence of attributes at the root level (might indicate layers)
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
            for key, value in f.attrs.items():
                print(f"- {key}: {value}")
    finally:
        f.close()  # Ensure proper file closure

# Example usage
weight_file = './weights'
print_structure(weight_file)