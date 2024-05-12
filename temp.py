import numpy as np
import matplotlib.pyplot as plt

import os

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.models import Model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.src.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Concatenate
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import Model
IMGWIDTH = 256